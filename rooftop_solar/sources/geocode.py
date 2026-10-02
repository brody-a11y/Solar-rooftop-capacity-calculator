"""Address -> coordinates.

auto:   Overture address points, then the Census geocoder if no match (default).
overture: Overture Maps address points (county/state address data). Free, US,
        needs a house number and ZIP; points usually sit on the parcel.
census: US Census Bureau geocoder. Free, no key, US only. Points are
        interpolated along the street, so they usually sit in the street in
        front of the building rather than on it.
google: Google Geocoding API. Billed; needs the Geocoding API enabled on the
        same Google Cloud key used for Solar. Often returns a rooftop point.
Results are cached on disk, so each address is looked up once.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from ..redact import redact

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
GOOGLE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


@dataclass
class GeocodeResult:
    lat: float
    lon: float
    source: str
    precision: str  # rooftop, interpolated, approximate
    matched_address: str


class Geocoder:
    def __init__(self, provider: str = "auto", api_key: str | None = None,
                 cache_path: str | Path = Path.home() / ".rooftop-solar" / "geocode_cache.json", timeout: float = 30.0,
                 addresses=None):
        if provider not in ("auto", "overture", "census", "google"):
            raise ValueError(f"unknown geocoder {provider!r}")
        if provider == "google" and not api_key:
            raise ValueError("google geocoding needs an API key")
        self.provider, self.api_key, self.timeout = provider, api_key, timeout
        self.cache_path = Path(cache_path)
        self._cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._lock = threading.Lock()
        self.session = requests.Session()
        self._addresses = addresses
        self.google_errors: list[str] = []  # Google failures this run (the free sources were used instead)

    @property
    def addresses(self):
        if self._addresses is None:
            from .overture_addresses import OvertureAddresses

            self._addresses = OvertureAddresses()
        return self._addresses

    def geocode(self, address: str) -> GeocodeResult | None:
        key = f"{self.provider}|{' '.join(address.lower().split())}"
        with self._lock:
            if key in self._cache:
                hit = self._cache[key]
                # A Census street-range estimate saved under the Google provider (from a
                # run where Google failed) is worth one more try with Google.
                if not (self.provider == "google" and hit and hit.get("source") == "census"):
                    return GeocodeResult(**hit) if hit else None
        google_failed = False
        if self.provider == "google":
            try:
                result = self._google(address)
            except Exception as exc:  # quota, key or network problem: fall back to the free sources
                with self._lock:
                    self.google_errors.append(redact(exc)[:200])
                result, google_failed = None, True
            if result is None:  # not found by Google (or Google failed): try Overture, then Census
                result = self._overture(address) or self._census(address)
        elif self.provider == "census":
            result = self._census(address)
        else:
            result = self._overture(address)
            if result is None and self.provider == "auto":
                result = self._census(address)
        if result is None and google_failed:
            return None  # don't remember "not found" when Google never answered; retry next run
        with self._lock:
            self._cache[key] = asdict(result) if result else None
        return result

    def save(self) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))

    def _overture(self, address: str) -> GeocodeResult | None:
        hit = self.addresses.lookup(address)
        if hit is None:
            return None
        lat, lon, matched, precision = hit
        return GeocodeResult(lat, lon, "overture", precision, matched)

    def _census(self, address: str) -> GeocodeResult | None:
        r = self.session.get(
            CENSUS_URL,
            params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        matches = r.json().get("result", {}).get("addressMatches", [])
        if not matches:
            return None
        m = matches[0]
        return GeocodeResult(m["coordinates"]["y"], m["coordinates"]["x"], "census", "interpolated", m.get("matchedAddress", ""))

    def _google(self, address: str) -> GeocodeResult | None:
        r = self.session.get(GOOGLE_URL, params={"address": address, "key": self.api_key}, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Google geocoding HTTP {r.status_code}: {r.text[:150]}")
        data = r.json()
        if data.get("status") == "ZERO_RESULTS":
            return None
        if data.get("status") != "OK":
            raise RuntimeError(f"Google geocoding: {data.get('status')} {data.get('error_message', '')}".strip())
        res = data["results"][0]
        loc_type = res["geometry"].get("location_type", "")
        precision = {"ROOFTOP": "rooftop", "RANGE_INTERPOLATED": "interpolated"}.get(loc_type, "approximate")
        loc = res["geometry"]["location"]
        return GeocodeResult(loc["lat"], loc["lng"], "google", precision, res.get("formatted_address", ""))
