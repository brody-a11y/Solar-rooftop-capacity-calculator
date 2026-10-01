"""Parcel boundaries from the Regrid Parcel API.

A parcel defines which buildings belong to a property, which the address alone
can't (garden-style complexes have many buildings and one address). Responses
are cached on disk; each uncached lookup uses one parcel record of the plan.

The trial token only covers a handful of sample counties; elsewhere Regrid
returns no parcels or an access error, reported per site.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from shapely.geometry import MultiPolygon, Point, Polygon, shape

API_V2_POINT = "https://app.regrid.com/api/v2/parcels/point"


@dataclass
class Parcel:
    geometry: Polygon | MultiPolygon  # lon/lat
    parcel_id: str
    address: str
    owner: str
    acres: float | None
    mail_address: str = ""  # owner's mailing address, for matching owners across parcels
    zip_code: str = ""  # situs (property) ZIP
    units: int | None = None  # dwelling/unit count where the county records it


_ENTITY_WORDS = {"LLC", "LP", "LLP", "INC", "CO", "CORP", "CORPORATION", "LTD", "THE", "OF", "A", "AN", "AND"}


def _norm_owner(owner: str) -> str:
    words = re.sub(r"[^A-Z0-9 ]", " ", owner.upper().replace(".", "")).split()
    return " ".join(w for w in words if w not in _ENTITY_WORDS)


def same_owner(a: Parcel, b: Parcel) -> bool:
    """Same owning entity: owner names equal after dropping LLC/LP/INC and
    punctuation, or the same owner mailing address (portfolio owners often hold
    each building in its own LLC but get their mail at one office)."""
    na, nb = _norm_owner(a.owner), _norm_owner(b.owner)
    if na and na == nb:
        return True
    ma, mb = _norm_owner(a.mail_address), _norm_owner(b.mail_address)
    return bool(ma) and ma == mb


def _features(data: dict) -> list[dict]:
    """GeoJSON features from a Regrid response, tolerating v1/v2 layouts."""
    for key in ("parcels", "results"):
        block = data.get(key)
        if isinstance(block, dict) and "features" in block:
            return block["features"]
        if isinstance(block, list):
            return block
    return data.get("features", []) if isinstance(data, dict) else []


def _parcel(feature: dict) -> Parcel | None:
    geom = feature.get("geometry")
    if not geom:
        return None
    g = shape(geom)
    if g.is_empty or not isinstance(g, (Polygon, MultiPolygon)):
        return None
    props = feature.get("properties") or {}
    fields = props.get("fields") or props
    acres = fields.get("ll_gisacre") or fields.get("gisacre")
    return Parcel(
        geometry=g if g.is_valid else g.buffer(0),
        parcel_id=str(fields.get("parcelnumb") or fields.get("ll_uuid") or props.get("headline") or ""),
        address=str(fields.get("address") or props.get("headline") or ""),
        owner=str(fields.get("owner") or ""),
        acres=float(acres) if acres not in (None, "") else None,
        mail_address=" ".join(str(fields.get(k) or "") for k in ("mailadd", "mail_zip")).strip(),
        zip_code=str(fields.get("szip5") or fields.get("szip") or "")[:5],
        units=_int(fields.get("numunits") or fields.get("units")),
    )


def _int(v) -> int | None:
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


class RegridError(RuntimeError):
    """Short reason code, e.g. http_403 or no_parcel_at_point."""


class RegridClient:
    def __init__(self, token: str, cache_dir: str | Path | None = Path.home() / ".rooftop-solar" / "regrid_cache",
                 timeout: float = 60.0, retries: int = 2):
        self.token = token
        self.retries = retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, params: dict):
        """GET with retries: Regrid's point lookup sometimes takes over 30 s."""
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(API_V2_POINT, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise RegridError("timeout") from None
            else:
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    return resp
            time.sleep(delay)
            delay *= 2
        raise RegridError("timeout")  # not reached

    def parcel_at(self, lat: float, lon: float) -> Parcel:
        """The parcel containing the point (or the nearest one Regrid returns)."""
        key = hashlib.sha1(f"{lat:.7f},{lon:.7f}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            data = json.loads(path.read_text())
        else:
            resp = self._get({"lat": f"{lat:.7f}", "lon": f"{lon:.7f}", "token": self.token})
            if resp.status_code != 200:
                raise RegridError(f"http_{resp.status_code}")
            data = resp.json()
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
        parcels = [p for p in (_parcel(f) for f in _features(data)) if p is not None]
        if not parcels:
            raise RegridError("no_parcel_at_point")
        pt = Point(lon, lat)
        containing = [p for p in parcels if p.geometry.contains(pt)]
        return containing[0] if containing else min(parcels, key=lambda p: p.geometry.distance(pt))
