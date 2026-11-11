"""Property name -> map point, via Google Places Text Search (New).

Street addresses often geocode to the street, a neighbour's lot or a leasing
office; a search for the property's name finds the place Google lists for the
complex, which usually sits on its own buildings. Billed per request (Places
API (New), Text Search with location), needs that API enabled on the same
Google Cloud key. Results are cached on disk, so each query is looked up once.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import requests

from ..redact import redact

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_FIELDS = "places.displayName,places.location,places.types,places.formattedAddress"


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    types: list[str]
    address: str


class PlacesClient:
    def __init__(self, api_key: str, cache_path: str | Path = Path.home() / ".rooftop-solar" / "places_cache.json",
                 timeout: float = 30.0):
        self.api_key, self.timeout = api_key, timeout
        self.cache_path = Path(cache_path)
        self._cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._lock = threading.Lock()
        self.session = requests.Session()
        self.errors: list[str] = []

    def search(self, query: str, lat: float, lon: float, radius_m: float = 2000.0) -> list[Place]:
        key = f"{' '.join(query.lower().split())}|{lat:.3f},{lon:.3f}"
        with self._lock:
            if key in self._cache:
                return [Place(**p) for p in self._cache[key]]
        body = {"textQuery": query, "maxResultCount": 5,
                "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": radius_m}}}
        headers = {"X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": _FIELDS}
        try:
            resp = self.session.post(SEARCH_URL, json=body, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # API not enabled, quota, network: report and carry on without it
            with self._lock:
                self.errors.append(redact(exc)[:200])
            return []
        places = [Place(p.get("displayName", {}).get("text", ""), p["location"]["latitude"], p["location"]["longitude"],
                        p.get("types", []), p.get("formattedAddress", ""))
                  for p in data.get("places", []) if "location" in p]
        with self._lock:
            self._cache[key] = [p.__dict__ for p in places]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))
        return places
