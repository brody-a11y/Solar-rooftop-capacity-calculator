"""Address -> coordinates.

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
    def __init__(self, provider: str = "census", api_key: str | None = None,
                 cache_path: str | Path = Path.home() / ".rooftop-solar" / "geocode_cache.json", timeout: float = 30.0):
        if provider not in ("census", "google"):
            raise ValueError(f"unknown geocoder {provider!r}")
        if provider == "google" and not api_key:
            raise ValueError("google geocoding needs an API key")
        self.provider, self.api_key, self.timeout = provider, api_key, timeout
        self.cache_path = Path(cache_path)
        self._cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._lock = threading.Lock()
        self.session = requests.Session()

    def geocode(self, address: str) -> GeocodeResult | None:
        key = f"{self.provider}|{' '.join(address.lower().split())}"
        with self._lock:
            if key in self._cache:
                hit = self._cache[key]
                return GeocodeResult(**hit) if hit else None
        result = self._census(address) if self.provider == "census" else self._google(address)
        with self._lock:
            self._cache[key] = asdict(result) if result else None
        return result

    def save(self) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))

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
        r.raise_for_status()
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
