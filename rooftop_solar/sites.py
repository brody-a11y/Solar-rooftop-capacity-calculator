"""Portfolio mode: a spreadsheet of addresses -> geocode -> footprints -> sizes.

Input is a CSV with an address column, or latitude/longitude columns, plus
optional id/name and occupancy columns. Header names are matched loosely
("Property Address", "Latitude", "lng", ...).
"""

from __future__ import annotations

import csv
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .calibration import Calibrator
from .geometry import LocalFrame, principal_axes
from .models import Building, Occupancy
from .pipeline import ReviewPolicy, SiteEstimate, estimate_many
from .sizing import GeometricEstimator
from .sources.geocode import Geocoder, GeocodeResult
from .sources.google_solar import GoogleInsights, GoogleSolarClient, footprint_from_insights
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
    "group": ("group", "property group", "parent property", "main address", "complex"),
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
    group: str = ""  # rows sharing a group are one property (e.g. one row per building)


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
            sites.append(Site(sid, address, lat, lon, _occupancy_from_text(get("occupancy")), get("group")))
    return sites


@dataclass
class SiteOutcome:
    site: Site
    geocode: GeocodeResult | None = None
    matches: list[FootprintMatch] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    estimates: list[SiteEstimate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    parcel: object | None = None  # regrid.Parcel when parcel lookup succeeded
    counted: list[bool] = field(default_factory=list)  # per estimate: included in the site total

    def _counted(self):
        flags = self.counted or [True] * len(self.estimates)
        return [e for e, c in zip(self.estimates, flags) if c]

    def _triples(self):
        flags = self.counted or [True] * len(self.estimates)
        return list(zip(self.buildings, self.estimates, flags))

    def raised_extra_kw(self, rooftop_only: bool = False) -> float:
        """kW added if raised racking spans equipment gaps (counted structures)."""
        total = 0.0
        for b, e, c in self._triples():
            if not c or not e.google or (rooftop_only and b.structure != "building"):
                continue
            mods = e.google.details.get("raised_racking_extra_modules", 0)
            per_module_kw = e.dc_kw / e.primary.module_count if e.primary and e.primary.module_count else 0.0
            total += mods * per_module_kw
        return total

    def rooftop_estimates(self):
        """Counted estimates on buildings only (no carports or garages)."""
        return [e for b, e, c in self._triples() if c and b.structure == "building"]

    @property
    def dc_kw(self) -> float:
        return sum(e.dc_kw for e in self._counted())

    def row(self) -> dict:
        g = self.geocode
        reasons = list(self.reasons)
        disagree = 0
        for b, e in zip(self.buildings, self.estimates):
            for r in e.reasons:
                if r.startswith("methods_disagree_ratio_"):
                    disagree += 1
                elif r not in reasons:
                    reasons.append(r)
        if disagree:  # one line, not one per building
            only = next(r for e in self.estimates for r in e.reasons if r.startswith("methods_disagree_ratio_"))
            reasons.append(only if disagree == 1 and len(self.estimates) == 1 else f"methods_disagree_on_{disagree}_of_{len(self.estimates)}_buildings")
        return {
            "site_id": self.site.id,
            "group": self.site.group,
            "address": self.site.address,
            "matched_address": g.matched_address if g else "",
            "lat": round(g.lat, 7) if g else "",
            "lon": round(g.lon, 7) if g else "",
            "location_source": f"{g.source}:{g.precision}" if g else "",
            "parcel_id": getattr(self.parcel, "parcel_id", ""),
            "parcel_acres": getattr(self.parcel, "acres", "") or "",
            "buildings": len(self.buildings),
            "dc_kw": round(self.dc_kw, 2),
            "raw_kw": round(sum(e.raw_kw for e in self._counted()), 2),
            "module_count": sum(e.primary.module_count for e in self._counted() if e.primary),
            "north_faces_kw_not_counted": round(sum(e.google.details.get("poleward_face_kw", 0) for e in self._counted()
                                                    if e.google and e.google.details.get("poleward_faces_excluded")), 1),
            "maxfit_raised_racking_kw": round(self.dc_kw + self.raised_extra_kw(), 2),
            "rooftop_kw": round(sum(e.dc_kw for b, e, c in self._triples() if c and b.structure == "building"), 2),
            "carport_kw": round(sum(e.dc_kw for b, e, c in self._triples() if c and b.structure != "building"), 2),
            "low_yield_kw_not_counted": round(sum(e.google.details.get("low_yield_kw", 0) for e in self._counted() if e.google), 1),
            "uncounted_buildings_kw": round(sum(e.dc_kw for e in self.estimates) - self.dc_kw, 1),
            "roof_area_m2": round(sum(e.primary.gross_roof_area_m2 for e in self._counted() if e.primary), 1),
            "needs_review": bool(reasons),
            "reasons": ";".join(reasons),
        }

    def building_rows(self) -> list[dict]:
        rows = []
        flags = self.counted or [True] * len(self.estimates)
        for b, m, e, c in zip(self.buildings, self.matches, self.estimates, flags):
            r = {"site_id": self.site.id, "counted": c, **e.row(b.occupancy.value)}
            r.update(
                structure=b.structure,
                overture_id=m.overture_id,
                overture_class=m.building_class or "",
                occupancy=b.occupancy.value,
                distance_from_address_m=round(m.distance_m, 1),
                height_m=m.height_m if m.height_m is not None else "",
                google_imagery_date=e.google.details.get("imagery_date", "") if e.google else "",
            )
            rows.append(r)
        return rows


_NON_ROOF_CLASSES = {"carport", "garage", "garages", "shed", "roof", "parking", "kiosk", "hut"}


def structure_kind(m: FootprintMatch, min_area_m2: float = 100.0, carport_width_m: float = 8.0,
                   carport_aspect: float = 3.5) -> str:
    """'building', or 'carport_or_garage' for structures rooftop-only designs leave
    out: tiny footprints, long narrow carport rows, or mapped as such."""
    if (m.building_class or "") in _NON_ROOF_CLASSES:
        return "carport_or_garage"
    local = LocalFrame.for_geometry(m.footprint).to_local(m.footprint)
    if local.area < min_area_m2:
        return "carport_or_garage"
    short, long_, _ = principal_axes(local)
    if short < carport_width_m and long_ / max(short, 1e-6) > carport_aspect:
        return "carport_or_garage"
    return "building"


def group_rows(outcomes: list["SiteOutcome"]) -> list[dict]:
    """One row per Group: summed sizes over its sites (e.g. one site per building)."""
    groups: dict[str, list[SiteOutcome]] = {}
    for o in outcomes:
        if o.site.group:
            groups.setdefault(o.site.group, []).append(o)
    rows = []
    for name, members in groups.items():
        member_rows = [m.row() for m in members]
        reasons: list[str] = []
        for r in member_rows:
            reasons.extend(x for x in r["reasons"].split(";") if x and x not in reasons)
        rows.append({
            "group": name,
            "sites": len(members),
            "buildings": sum(r["buildings"] for r in member_rows),
            "dc_kw": round(sum(r["dc_kw"] for r in member_rows), 2),
            "raw_kw": round(sum(r["raw_kw"] for r in member_rows), 2),
            "module_count": sum(r["module_count"] for r in member_rows),
            "needs_review": any(r["needs_review"] for r in member_rows),
            "reasons": ";".join(reasons),
        })
    return rows


def _google_footprint(o: SiteOutcome, client: GoogleSolarClient, min_panels: int = 40,
                      ring_m: float = 25.0, reach_m: float = 25.0) -> FootprintMatch | None:
    """Google's building at the site's point, for buildings the footprint data
    doesn't have yet (usually new construction).

    Address pins often sit on a sidewalk or entry canopy, so if the building at
    the pin is small, four more points 25 m away (N, E, S, W) are tried, and the
    largest building with panels within `reach_m` of the pin is used.
    """
    frame = LocalFrame(o.geocode.lon, o.geocode.lat)
    best = None
    offsets = [(0.0, 0.0), (0.0, ring_m), (ring_m, 0.0), (0.0, -ring_m), (-ring_m, 0.0)]
    for i, (dx, dy) in enumerate(offsets):
        lon, lat = frame.point_to_lonlat(dx, dy)
        try:
            ins = GoogleInsights.from_response(client.building_insights(lat, lon))
        except Exception:
            continue
        if len(ins.panels) < min_panels:
            continue
        near = min(math.hypot(*frame.point_to_local(p.lon, p.lat)) for p in ins.panels)
        if near <= reach_m and (best is None or len(ins.panels) > len(best.panels)):
            best = ins
        if i == 0 and best is not None:
            break  # the pin itself is on a real building
    if best is None:
        return None
    fp = footprint_from_insights(best)
    if fp is None:
        return None
    return FootprintMatch(f"google:{best.name}", fp, 0.0, None, None, None, None, None)


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
    google_max_points: int = 9,
    unit_addresses=None,
    parcels=None,
    max_parcel_buildings: int = 120,
    include_carports: bool = True,
    carport_min_energy_ratio: float = 0.8,
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

    # Multi-building properties, best source first: the parcel (every building on it).
    parcel_sites: set[str] = set()
    if parcels is not None:
        progress(f"Looking up parcels for {len(located)} sites...")
        areas = {}
        for o in outcomes:
            if not o.geocode:
                continue
            try:
                parcel = parcels.parcel_at(o.geocode.lat, o.geocode.lon)
            except Exception as exc:  # parcel data is a refinement; fall back to the address match
                o.reasons.append(f"parcel_lookup_failed:{exc}")
                continue
            o.parcel = parcel
            areas[o.site.id] = parcel.geometry
        on_parcel = footprints.in_polygons(areas) if areas else {}
        for sid, ms in on_parcel.items():
            o = next(x for x in outcomes if x.site.id == sid)
            if not include_carports:
                kept = [m for m in ms if structure_kind(m) == "building"]
                if len(kept) < len(ms):
                    o.reasons.append(f"skipped_{len(ms) - len(kept)}_carport_or_garage_structures")
                ms = kept
            if not ms:
                o.reasons.append("no_mapped_buildings_on_parcel")
                continue
            if len(ms) > max_parcel_buildings:
                o.reasons.append(f"parcel_has_{len(ms)}_buildings_kept_largest_{max_parcel_buildings}_check")
                ms = ms[:max_parcel_buildings]
            found[sid] = ms
            parcel_sites.add(sid)
            o.reasons.append(f"parcel_{o.parcel.parcel_id or 'unknown'}_{len(ms)}_buildings")

    # Otherwise: buildings under the county's per-unit address points.
    if unit_addresses is not None and campus_m == 0:
        unit_pts: dict[str, tuple[float, float]] = {}
        for o in outcomes:
            if not (o.geocode and o.site.address and found.get(o.site.id)) or o.site.id in parcel_sites:
                continue
            try:
                pts = unit_addresses.unit_points(o.site.address)
            except Exception:  # unit points are a bonus; never fail the site over them
                pts = []
            if len(pts) > 1:
                for i, (lat, lon) in enumerate(pts):
                    unit_pts[f"{o.site.id}\x00{i}"] = (lon, lat)
        if unit_pts:
            progress(f"Adding buildings under {len(unit_pts)} unit address points...")
            extra = footprints.find(unit_pts, search_m=3.0)
            for key, ms in extra.items():
                sid = key.split("\x00")[0]
                have = {m.overture_id for m in found[sid]}
                for m in ms[:1]:
                    if m.overture_id not in have and m.distance_m <= 3.0:
                        found[sid].append(m)
                        have.add(m.overture_id)

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
            fallback = _google_footprint(o, google_client) if google_client else None
            if fallback is None:
                o.reasons.append(f"no_building_within_{search_m:.0f}m")
                continue
            o.matches = [fallback]
            o.reasons.append("footprint_from_google_imagery_check_date")
        m0 = o.matches[0]
        # Street-level points (Census) normally land a few metres off the building;
        # only flag matches that are far away or nearly tied with another building.
        # Parcel matches don't depend on the point, so they skip this check.
        if o.site.id in parcel_sites:
            pass
        elif m0.distance_m > far_m:
            o.reasons.append(f"address_point_{m0.distance_m:.0f}m_from_building_check_match")
        elif m0.distance_m > 0 and m0.runner_up_m is not None and m0.runner_up_m - m0.distance_m < tie_m:
            o.reasons.append("two_buildings_equally_close_check_match")
        if len(o.matches) > 1 and o.site.id not in parcel_sites:
            source = "unit_address_points" if campus_m == 0 else f"{campus_m:.0f}m_radius"
            o.reasons.append(f"campus_{len(o.matches)}_buildings_from_{source}")
        for n, m in enumerate(o.matches):
            occ = o.site.occupancy or occupancy_from_class(m.building_class)
            if occ is None:
                occ = Occupancy.R2
                if "occupancy_assumed_multifamily" not in o.reasons:
                    o.reasons.append("occupancy_assumed_multifamily")
            bid = o.site.id if len(o.matches) == 1 else f"{o.site.id} #{n + 1}"
            kind = structure_kind(m) if o.site.id in parcel_sites else "building"
            o.buildings.append(Building(bid, m.footprint, occ, structure=kind))
            if m.roof_shape and m.roof_shape not in ("flat",) and google_client is None:
                o.reasons.append(f"mapped_roof_shape_{m.roof_shape}_but_sized_as_flat")

    # 3. sizing
    jobs = [(o, b) for o in outcomes for b in o.buildings]
    progress(f"Sizing {len(jobs)} buildings...")
    results = estimate_many([b for _o, b in jobs], geometric, google_client, calibrator, policy, workers, google_max_points)
    for (o, _b), est in zip(jobs, results):
        o.estimates.append(est)
    for o in outcomes:
        o.counted = [True] * len(o.estimates)
        if o.site.id not in parcel_sites or len(o.estimates) < 2:
            continue
        with_google = [e.method == "google_filtered" for e in o.estimates]
        if any(with_google) and not all(with_google):
            # Outline-only sizing ignores rooftop equipment and runs ~2-3x high; on a
            # parcel Google otherwise covers, such structures are reported, not counted.
            o.counted = list(with_google)
            o.reasons.append(f"not_counted_{with_google.count(False)}_buildings_without_google_data")
        # Carports and garages count only when Google sized them (so shading is known)
        # and their typical panel yields >= carport_min_energy_ratio of the best roof panel.
        best = max((e.google.details.get("best_panel_kwh", 0.0) for b, e in zip(o.buildings, o.estimates)
                    if e.google and b.structure == "building"), default=0.0)
        shaded = no_data = 0
        for i, (b, e) in enumerate(zip(o.buildings, o.estimates)):
            if b.structure == "building" or not o.counted[i]:
                continue
            if e.method != "google_filtered":
                o.counted[i] = False
                no_data += 1
            elif best and e.google.details.get("median_panel_kwh", 0.0) < carport_min_energy_ratio * best:
                o.counted[i] = False
                shaded += 1
        if shaded:
            o.reasons.append(f"not_counted_{shaded}_shaded_carports")
        if no_data and not any("without_google_data" in r for r in o.reasons):
            o.reasons.append(f"not_counted_{no_data}_carports_without_google_data")
    return outcomes
