"""Installed rooftop PV systems from city building-permit open data (Socrata).

Permits give an address and, in the work description, usually the system size
("INSTALL 250 KW ROOFTOP PV SYSTEM"). Installed systems are sized to budget,
load or interconnection limits, so they are minimums for MaxFit ("floor" truth
in the accuracy test), and their ratio to MaxFit shows how much of a roof
typically gets built.

Column names differ by city, so datasets are found through the Socrata
catalog and their columns are matched by name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

CATALOG_URL = "https://api.us.socrata.com/api/catalog/v1"

# City portals with building permits on Socrata. Name -> (domain, city, state).
CITIES = {
    "sf": ("data.sfgov.org", "San Francisco", "CA"),
    "la": ("data.lacity.org", "Los Angeles", "CA"),
    "austin": ("data.austintexas.gov", "Austin", "TX"),
    "seattle": ("data.seattle.gov", "Seattle", "WA"),
    "chicago": ("data.cityofchicago.org", "Chicago", "IL"),
    "nyc": ("data.cityofnewyork.us", "New York", "NY"),
}

_KW = re.compile(r"(\d{1,4}(?:[.,]\d+)?)\s*-?\s*(?:KW|KILO\s*WATTS?)(?:\s*-?\s*DC)?\b", re.I)
_MW = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*-?\s*MW\b", re.I)
_NOT_ROOF = re.compile(r"CARPORT|CANOPY|GROUND[\s-]*MOUNT|PARKING\s+STRUCTURE|TRELLIS|SHADE\s+STRUCTURE|PERGOLA", re.I)
_NOT_NEW = re.compile(r"\bREMOVE\b|DETACH|RE-?INSTALL|REVISION TO|BATTERY ONLY|ESS ONLY|EV CHARG", re.I)
_SOLAR = re.compile(r"SOLAR|PHOTOVOLTAIC|\bPV\b", re.I)


def kw_from_text(text: str) -> float | None:
    """System size in kW from a permit description, or None. Takes the largest
    figure stated (descriptions often list module and inverter ratings too)."""
    vals = [float(m.group(1).replace(",", ".")) for m in _KW.finditer(text or "")]
    vals += [1000 * float(m.group(1)) for m in _MW.finditer(text or "")]
    vals = [v for v in vals if 0 < v < 20000]
    return max(vals) if vals else None


def rooftop_new_install(text: str) -> bool:
    return bool(_SOLAR.search(text or "")) and not _NOT_ROOF.search(text) and not _NOT_NEW.search(text)


def _col(columns: list[str], *patterns: str) -> str | None:
    for pat in patterns:
        for c in columns:
            if re.fullmatch(pat, c, re.I):
                return c
    for pat in patterns:
        for c in columns:
            if re.search(pat, c, re.I):
                return c
    return None


@dataclass
class PermitDataset:
    domain: str
    id: str
    name: str
    columns: list[str]

    @property
    def description(self) -> str | None:
        return _col(self.columns, r"(work_)?description", r"job_description", r"desc", r"scope", r"work")


def find_datasets(session: requests.Session, domain: str, timeout: float = 60) -> list[PermitDataset]:
    """Permit datasets on a Socrata portal that have a description column."""
    resp = session.get(CATALOG_URL, params={"domains": domain, "search_context": domain, "q": "building permits",
                                            "only": "datasets", "limit": 20}, timeout=timeout)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        res = r.get("resource") or {}
        ds = PermitDataset(domain, res.get("id", ""), res.get("name", ""), list(res.get("columns_field_name") or []))
        if ds.id and re.search(r"permit", ds.name, re.I) and ds.description:
            out.append(ds)
    return out


def _address(row: dict, columns: list[str]) -> str:
    one = _col(columns, r"(original_)?address1?", r"(street_)?address", r"location_address", r"address")
    if one and isinstance(row.get(one), str) and re.match(r"\s*\d", row[one]):
        return row[one].strip()
    parts = [_col(columns, r"street_?(number|no|num)", r"house_?(number|no)", r"house__?"),
             _col(columns, r"street_?(direction|dir|prefix)"),
             _col(columns, r"street_?name"),
             _col(columns, r"street_?(suffix|type)")]
    return " ".join(str(row.get(p) or "").strip() for p in parts if p and row.get(p)).strip()


def _latlon(row: dict, columns: list[str]) -> tuple[float | None, float | None]:
    lat_c, lon_c = _col(columns, r"latitude", r"lat"), _col(columns, r"longitude", r"lon|lng")
    try:
        if lat_c and lon_c and row.get(lat_c) and row.get(lon_c):
            return float(row[lat_c]), float(row[lon_c])
        for c in columns:
            v = row.get(c)
            if isinstance(v, dict) and v.get("type") == "Point":
                lon, lat = v["coordinates"][:2]
                return float(lat), float(lon)
            if isinstance(v, dict) and v.get("latitude"):
                return float(v["latitude"]), float(v["longitude"])
    except (TypeError, ValueError, KeyError):
        pass
    return None, None


def pull_permits(city: str, min_kw: float = 30.0, since_year: int = 2015, limit: int = 2000,
                 session: requests.Session | None = None, timeout: float = 90) -> list[dict]:
    """Rooftop PV permits of at least `min_kw` (commercial/multifamily scale), one
    row per address (largest system), as truth rows of kind "floor"."""
    domain, city_name, state = CITIES[city]
    session = session or requests.Session()
    rows: dict[str, dict] = {}
    for ds in find_datasets(session, domain, timeout):
        desc = ds.description
        where = " OR ".join(f"upper({desc}) like '%{w}%'" for w in ("SOLAR", "PHOTOVOLTAIC", " PV "))
        date_c = _col(ds.columns, r"issue(d)?_date", r"issue_?date", r"permit_?issue", r"application.*date", r"date")
        params = {"$where": f"({where})", "$limit": str(limit)}
        if date_c:
            params["$order"] = f"{date_c} DESC"
        resp = session.get(f"https://{domain}/resource/{ds.id}.json", params=params, timeout=timeout)
        if resp.status_code >= 400:
            continue
        zip_c = _col(ds.columns, r"(original_)?zip(code)?", r"zip")
        id_c = _col(ds.columns, r"permit_?(number|no|num)", r"permit_?id", r"job__?", r"permit")
        for row in resp.json():
            text = str(row.get(desc) or "")
            kw = kw_from_text(text)
            if kw is None or kw < min_kw or not rooftop_new_install(text):
                continue
            date = str(row.get(date_c) or "")[:10] if date_c else ""
            if date and date[:4].isdigit() and int(date[:4]) < since_year:
                continue
            street = _address(row, ds.columns)
            if not street or not re.match(r"\d", street):
                continue
            zipcode = str(row.get(zip_c) or "")[:5] if zip_c else ""
            address = f"{street}, {city_name}, {state}" + (f" {zipcode}" if zipcode else "")
            key = re.sub(r"\W+", " ", street.upper()).strip()
            lat, lon = _latlon(row, ds.columns)
            rec = {"name": f"{city_name} {street.title()}", "address": address,
                   "latitude": lat if lat is not None else "", "longitude": lon if lon is not None else "",
                   "true_kw": kw, "module_w": "", "kind": "floor",
                   "source": f"{city} permit {row.get(id_c, '') if id_c else ''} {date} ({ds.name})".strip(),
                   "description": text[:300]}
            if key not in rows or kw > rows[key]["true_kw"]:
                rows[key] = rec
    return list(rows.values())
