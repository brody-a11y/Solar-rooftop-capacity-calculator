"""Portfolio mode: a spreadsheet of addresses -> geocode -> footprints -> sizes.

Input is a CSV with an address column, or latitude/longitude columns, plus
optional id/name and occupancy columns. Header names are matched loosely
("Property Address", "Latitude", "lng", ...).
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .calibration import Calibrator
from .models import Building, Occupancy
from .pipeline import ReviewPolicy, SiteEstimate, estimate_many
from .sizing import GeometricEstimator
from .sources.geocode import Geocoder, GeocodeResult
from .sources.google_solar import GoogleSolarClient
from .sources.overture import FootprintMatch, OvertureFootprints, occupancy_from_class

_HEADERS = {
    "id": ("id", "site id", "property id", "building id", "name", "property name", "property", "site"),
    "address": ("address", "full address", "property address", "site address", "location"),
    "street": ("street", "address line 1", "address1", "street address"),
    "city": ("city",),
    "state": ("state", "st"),
    "zip": ("zip", "zip code", "zipcode", "postal code", "postcode"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "lng", "long", "longitude", "x"),
    "occupancy": ("occupancy", "building type", "type", "property type"),
}


def _norm(h: str) -> str:
    return " ".join(h.strip().lower().replace("_", " ").split())


def _occupancy_from_text(text: str) -> Occupancy | None:
    t = text.strip().lower()
    if not t:
        return None
    if "commercial" in t or t in ("office", "retail", "industrial", "warehouse"):
        return Occupancy.COMMERCIAL
    if "multi" in t or "apartment" in t or t in ("r-2", "r2", "mf"):
        return Occupancy.R2
    if t in ("r-3", "r3") or "townho" in t or "single" in t:
        return Occupancy.R3
    return None


@dataclass
class Site:
    id: str
    address: str = ""
    lat: float | None = None
    lon: float | None = None
    occupancy: Occupancy | None = None


def read_sites(path: str) -> list[Site]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols: dict[str, str] = {}
        for h in reader.fieldnames or []:
            n = _norm(h)
            for key, names in _HEADERS.items():
                if n in names and key not in cols:
                    cols[key] = h
        has_ll = "lat" in cols and "lon" in cols
        if not ("address" in cols or "street" in cols or has_ll):
            raise ValueError(
                f"{path}: needs an 'address' column (or street/city/state/zip, or latitude/longitude). "
                f"Found: {', '.join(reader.fieldnames or [])}"
            )
        sites, seen = [], {}
        for i, row in enumerate(reader, 1):
            get = lambda k: (row.get(cols[k]) or "").strip() if k in cols else ""
            address = get("address") or get("street")
            city, state_zip = get("city"), " ".join(filter(None, (get("state"), get("zip"))))
            # append separate city/state/zip columns unless the address already has them
            if address and city and city.lower() not in address.lower():
                address = ", ".join(filter(None, (address, city, state_zip)))
            elif address and state_zip and state_zip.split()[0].lower() not in address.lower():
                address = f"{address}, {state_zip}"
            lat = lon = None
            if has_ll and get("lat") and get("lon"):
                lat, lon = float(get("lat")), float(get("lon"))
            if not address and lat is None:
                continue
            sid = get("id") or f"site-{i}"
            seen[sid] = seen.get(sid, 0) + 1
            if seen[sid] > 1:
                sid = f"{sid} ({seen[sid]})"
            sites.append(Site(sid, address, lat, lon, _occupancy_from_text(get("occupancy"))))
    return sites


@dataclass
class SiteOutcome:
    site: Site
    geocode: GeocodeResult | None = None
    matches: list[FootprintMatch] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    estimates: list[SiteEstimate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def dc_kw(self) -> float:
        return sum(e.dc_kw for e in self.estimates)

    def row(self) -> dict:
        g = self.geocode
        reasons = list(self.reasons)
        for b, e in zip(self.buildings, self.estimates):
            reasons.extend(r for r in e.reasons if r not in reasons)
        return {
            "site_id": self.site.id,
            "address": self.site.address,
            "matched_address": g.matched_address if g else "",
            "lat": round(g.lat, 7) if g else "",
            "lon": round(g.lon, 7) if g else "",
            "location_source": f"{g.source}:{g.precision}" if g else "",
            "buildings": len(self.buildings),
            "dc_kw": round(self.dc_kw, 2),
            "raw_kw": round(sum(e.raw_kw for e in self.estimates), 2),
            "module_count": sum(e.primary.module_count for e in self.estimates if e.primary),
            "roof_area_m2": round(sum(e.primary.gross_roof_area_m2 for e in self.estimates if e.primary), 1),
            "needs_review": bool(reasons),
            "reasons": ";".join(reasons),
        }

    def building_rows(self) -> list[dict]:
        rows = []
        for b, m, e in zip(self.buildings, self.matches, self.estimates):
            r = {"site_id": self.site.id, **e.row(b.occupancy.value)}
            r.update(
                overture_id=m.overture_id,
                overture_class=m.building_class or "",
                occupancy=b.occupancy.value,
                distance_from_address_m=round(m.distance_m, 1),
                height_m=m.height_m if m.height_m is not None else "",
            )
            rows.append(r)
        return rows


def size_sites(
    sites: list[Site],
    footprints: OvertureFootprints,
    geometric: GeometricEstimator,
    geocoder: Geocoder | None = None,
    google_client: GoogleSolarClient | None = None,
    calibrator: Calibrator | None = None,
    policy: ReviewPolicy = ReviewPolicy(),
    search_m: float = 40.0,
    campus_m: float = 0.0,
    workers: int = 8,
    progress=print,
    far_m: float = 25.0,
    tie_m: float = 10.0,
) -> list[SiteOutcome]:
    outcomes = [SiteOutcome(s) for s in sites]

    # 1. locations
    def locate(o: SiteOutcome):
        s = o.site
        if s.lat is not None:
            o.geocode = GeocodeResult(s.lat, s.lon, "input", "as_given", "")
            return
        if geocoder is None:
            o.reasons.append("no_coordinates_and_no_geocoder")
            return
        try:
            o.geocode = geocoder.geocode(s.address)
        except Exception as exc:  # one bad address must not stop a portfolio
            o.reasons.append(f"geocode_error:{type(exc).__name__}")
            return
        if o.geocode is None:
            o.reasons.append("address_not_found")

    progress(f"Locating {len(sites)} sites...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(locate, outcomes))
    if geocoder:
        geocoder.save()

    # 2. footprints
    located = {o.site.id: (o.geocode.lon, o.geocode.lat) for o in outcomes if o.geocode}
    progress(f"Finding building footprints for {len(located)} sites (first run builds an index, ~1 min)...")
    found = footprints.find(located, search_m=search_m, campus_m=campus_m) if located else {}

    owner: dict[str, str] = {}
    for o in outcomes:
        if not o.geocode:
            continue
        o.matches = found.get(o.site.id, [])
        for m in o.matches:
            if m.overture_id in owner:
                o.reasons.append(f"same_building_as_site_{owner[m.overture_id]}")
            else:
                owner[m.overture_id] = o.site.id
        if not o.matches:
            o.reasons.append(f"no_building_within_{search_m:.0f}m")
            continue
        m0 = o.matches[0]
        # Street-level points (Census) normally land a few metres off the building;
        # only flag matches that are far away or nearly tied with another building.
        if m0.distance_m > far_m:
            o.reasons.append(f"address_point_{m0.distance_m:.0f}m_from_building_check_match")
        elif m0.distance_m > 0 and m0.runner_up_m is not None and m0.runner_up_m - m0.distance_m < tie_m:
            o.reasons.append("two_buildings_equally_close_check_match")
        if len(o.matches) > 1:
            o.reasons.append(f"campus_{len(o.matches)}_buildings")
        for n, m in enumerate(o.matches):
            occ = o.site.occupancy or occupancy_from_class(m.building_class)
            if occ is None:
                occ = Occupancy.R2
                if "occupancy_assumed_multifamily" not in o.reasons:
                    o.reasons.append("occupancy_assumed_multifamily")
            bid = o.site.id if len(o.matches) == 1 else f"{o.site.id} #{n + 1}"
            o.buildings.append(Building(bid, m.footprint, occ))
            if m.roof_shape and m.roof_shape not in ("flat",) and google_client is None:
                o.reasons.append(f"mapped_roof_shape_{m.roof_shape}_but_sized_as_flat")

    # 3. sizing
    jobs = [(o, b) for o in outcomes for b in o.buildings]
    progress(f"Sizing {len(jobs)} buildings...")
    results = estimate_many([b for _o, b in jobs], geometric, google_client, calibrator, policy, workers)
    for (o, _b), est in zip(jobs, results):
        o.estimates.append(est)
    return outcomes
