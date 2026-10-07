"""Portfolio mode: a spreadsheet of addresses -> geocode -> footprints -> sizes.

Input is a CSV with an address column, or latitude/longitude columns, plus
optional id/name and occupancy columns. Header names are matched loosely
("Property Address", "Latitude", "lng", ...).
"""

from __future__ import annotations

import csv
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from shapely.geometry import Point

from .calibration import Calibrator
from .geometry import LocalFrame, principal_axes
from .models import Building, Occupancy
from .pipeline import ReviewPolicy, SiteEstimate, estimate_many
from .redact import redact
from .sizing import GeometricEstimator
from .sources.geocode import Geocoder, GeocodeResult
from .sources.google_solar import GoogleInsights, GoogleSolarClient, footprint_from_insights
from .sources.overture import FootprintMatch, OvertureFootprints, occupancy_from_class
from .sources.regrid import same_owner

_HEADERS = {
    "id": ("id", "site id", "property id", "building id", "name", "property name", "property", "site"),
    "address": ("address", "full address", "property address", "site address", "location"),
    "street": ("street", "address line 1", "address1", "street address"),
    "city": ("city",),
    "state": ("state", "st"),
    "zip": ("zip", "zip code", "zipcode", "postal code", "postcode"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "lng", "long", "longitude", "x"),
    "occupancy": ("occupancy", "building type", "type", "property type", "housing type", "housingtype"),
    "units": ("units", "unit count", "number of units", "# units", "num units", "total units"),
    "group": ("group", "property group", "parent property", "main address", "complex"),
    "built": ("built", "year built", "yearbuilt", "built year", "year completed", "completed"),
}


def _norm(h: str) -> str:
    return " ".join(h.strip().lower().replace("_", " ").split())


def _occupancy_from_text(text: str) -> Occupancy | None:
    t = text.strip().lower()
    if not t:
        return None
    if "commercial" in t or t in ("office", "retail", "industrial", "warehouse"):
        return Occupancy.COMMERCIAL
    # Single-family wording first: "Single-Family Home Apartments" and "Attached
    # Townhomes" are build-to-rent homes, not apartment buildings.
    if t in ("r-3", "r3") or "townho" in t or "single" in t or "btr" in t or "rental home" in t:
        return Occupancy.R3
    if "multi" in t or "apartment" in t or t in ("r-2", "r2", "mf"):
        return Occupancy.R2
    return None


@dataclass
class Site:
    id: str
    address: str = ""
    lat: float | None = None
    lon: float | None = None
    occupancy: Occupancy | None = None
    group: str = ""  # rows sharing a group are one property (e.g. one row per building)
    units: int | None = None  # homes/apartments at the property, if the input lists them
    year_built: int | None = None


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
                try:
                    lat, lon = float(get("lat").replace(",", "")), float(get("lon").replace(",", ""))
                except ValueError:
                    print(f"Row {i}: latitude/longitude '{get('lat')}', '{get('lon')}' aren't numbers; "
                          + ("using the address instead." if address else "row skipped."))
            if not address and lat is None:
                continue
            sid = get("id") or f"site-{i}"
            seen[sid] = seen.get(sid, 0) + 1
            if seen[sid] > 1:
                sid = f"{sid} ({seen[sid]})"
            units = re.sub(r"[^0-9.]", "", get("units"))
            built = re.search(r"\b(1[89]\d\d|20\d\d)\b", get("built"))
            sites.append(Site(sid, address, lat, lon, _occupancy_from_text(get("occupancy")), get("group"),
                              int(float(units)) if units and float(units) > 0 else None,
                              int(built.group(1)) if built else None))
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
    manual_review: list[str] = field(default_factory=list)  # low-confidence reasons a person should look at
    manual_review_extra: list[str] = field(default_factory=list)  # review reasons found while sizing
    listed_units: int | None = None  # unit address records found for the site's address
    parcel_units: int | None = None  # units in the county parcel records (Regrid), where recorded
    floor_m2_per_unit: float | None = None  # floor area of the buildings found, per input unit
    units_estimate_kw: float | None = None  # units x typical kW/unit, when the buildings found are too few

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
            "units_in_input": self.site.units or "",
            "kw_per_unit": round(self.dc_kw / self.site.units, 2) if self.site.units else "",
            "units_in_address_data": self.listed_units or "",
            "units_in_parcel_records": self.parcel_units or "",
            "dc_kw": round(self.dc_kw, 2),
            "best_estimate_kw": round(max(self.dc_kw, self.units_estimate_kw or 0.0), 2),
            "estimate_basis": "units" if self.units_estimate_kw and self.units_estimate_kw > self.dc_kw else "roof",
            "kw_estimate_from_units": round(self.units_estimate_kw, 1) if self.units_estimate_kw else "",
            "floor_m2_per_unit": round(self.floor_m2_per_unit, 1) if self.floor_m2_per_unit is not None else "",
            "year_built": self.site.year_built or "",
            "existing_solar_check": existing_solar_check(self),
            "raw_kw": round(sum(e.raw_kw for e in self._counted()), 2),
            "module_count": sum(e.primary.module_count for e in self._counted() if e.primary),
            "north_faces_kw_not_counted": round(sum(e.google.details.get("poleward_face_kw", 0) for e in self._counted()
                                                    if e.google and e.google.details.get("poleward_faces_excluded")), 1),
            "maxfit_raised_racking_kw": round(self.dc_kw + self.raised_extra_kw(), 2),
            "rooftop_kw": round(sum(e.dc_kw for b, e, c in self._triples() if c and b.structure == "building"), 2),
            "carport_kw": round(sum(e.dc_kw for b, e, c in self._triples() if c and b.structure != "building"), 2),
            "equipment_clearance_kw_not_counted": round(sum(e.google.details.get("equipment_clearance_kw", 0)
                                                            for e in self._counted() if e.google), 1),
            "low_yield_kw_not_counted": round(sum(e.google.details.get("low_yield_kw", 0) for e in self._counted() if e.google), 1),
            "uncounted_buildings_kw": round(sum(e.dc_kw for e in self.estimates) - self.dc_kw, 1),
            "roof_area_m2": round(sum(e.primary.gross_roof_area_m2 for e in self._counted() if e.primary), 1),
            "manual_review": bool(self.manual_review),
            "manual_review_reason": ";".join(self.manual_review),
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


_NONRESIDENTIAL_CLASSES = {"retail", "commercial", "warehouse", "industrial", "office", "supermarket", "manufacture",
                           "service", "parking", "school", "hospital", "church"}
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
        # Rows of one group can match the same building (e.g. two addresses on one
        # parcel): count each building once.
        seen, buildings, dc, raw, mods = set(), 0, 0.0, 0.0, 0
        for m in members:
            for (b, e, c), match in zip(m._triples(), m.matches or [None] * len(m.estimates)):
                key = match.overture_id if match is not None else (m.site.id, b.id)
                if key in seen:
                    continue
                seen.add(key)
                buildings += 1
                if c:
                    dc += e.dc_kw
                    raw += e.raw_kw
                    mods += e.primary.module_count if e.primary else 0
        rows.append({
            "group": name,
            "sites": len(members),
            "buildings": buildings,
            "dc_kw": round(dc, 2),
            "raw_kw": round(raw, 2),
            "module_count": mods,
            "manual_review": any(r["manual_review"] for r in member_rows),
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
    carport_min_kw: float = 15.0,
    max_owner_lookups: int = 10,
    community_sample: int = 6,
    community_owner_checks: bool = False,
    min_kw_per_unit: float = 0.1,
    big_building_m2: float = 800.0,
    max_kw_per_unit_one_building: float = 4.0,
    equipment_client=None,
    stale_imagery_years: float = 6.0,
    min_imagery_coverage: float = 0.25,
    outline_only_factor: float = 0.55,
    min_floor_m2_per_unit: float = 15.0,
    kw_per_unit_fallback: float = 1.7,
    review_imagery_flags: bool = False,
    today: date | None = None,
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
            detail = re.sub(r"[^A-Za-z0-9_.*-]+", "_", redact(exc))[:60].strip("_")
            o.reasons.append(f"geocode_error:{type(exc).__name__}" + (f"_{detail}" if detail else ""))
            return
        if o.geocode is None:
            o.reasons.append("address_not_found")

    progress(f"Locating {len(sites)} sites...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(locate, outcomes))
    if geocoder:
        geocoder.save()
        from collections import Counter

        found = Counter(o.geocode.source for o in outcomes if o.geocode)
        missing = sum(1 for o in outcomes if not o.geocode)
        line = f"Address lookup: {sum(found.values())} found ({', '.join(f'{k} {v}' for k, v in found.items()) or 'none'})"
        line += f", {missing} not found" if missing else ""
        errs = Counter(getattr(geocoder, "google_errors", []))
        if errs:
            msg, n = errs.most_common(1)[0]
            line += f"\n  Google geocoding failed {sum(errs.values())} times; used free sources instead. Google said: {msg}"
        progress(line)

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
            if community_sample > 0 and not community_owner_checks and o.site.occupancy == Occupancy.R3 and o.site.units:
                continue  # single-family community: sized from a sample of homes, no parcel records needed
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

    # Unit address points for each site. Input addresses often lack a ZIP, which
    # the address data is keyed on: take it from the geocoder's matched address
    # or the parcel record.
    site_units: dict[str, list[tuple[float, float]]] = {}
    if unit_addresses is not None:
        for o in outcomes:
            addr = _address_with_zip(o)
            if not addr:
                continue
            try:
                if hasattr(unit_addresses, "address_records"):
                    records = unit_addresses.address_records(addr)
                    site_units[o.site.id] = sorted({(round(r["lat"], 6), round(r["lon"], 6)) for r in records})
                    o.listed_units = len({(r.get("number"), r["unit"]) for r in records if r.get("unit")}) or None
                else:
                    site_units[o.site.id] = unit_addresses.unit_points(addr)
            except Exception:  # unit points are a bonus; never fail the site over them
                continue
            if addr != o.site.address:
                o.reasons.append(f"zip_{addr.rsplit(' ', 1)[-1]}_added_for_unit_lookup")
    for o in outcomes:
        if o.parcel is not None:
            o.parcel_units = o.parcel.units

    # Complexes often span several parcels, one per building or phase. Follow the
    # site's unit address points onto neighbouring parcels and keep those with the
    # same owner (name or mailing address).
    if parcels is not None and unit_addresses is not None:
        for o in outcomes:
            if o.site.id not in parcel_sites:
                continue
            pts = site_units.get(o.site.id, [])
            seen = [o.parcel.geometry]
            extra_areas, lookups, other_owner = {}, 0, 0
            for lat, lon in pts:
                if lookups >= max_owner_lookups or any(g.covers(Point(lon, lat)) for g in seen):
                    continue
                lookups += 1
                try:
                    p = parcels.parcel_at(lat, lon)
                except Exception:
                    continue
                seen.append(p.geometry)
                if same_owner(p, o.parcel):
                    extra_areas[f"{o.site.id}\x00{len(extra_areas)}"] = p.geometry
                    if p.units:
                        o.parcel_units = (o.parcel_units or 0) + p.units
                else:
                    other_owner += 1
            if extra_areas:
                have = {m.overture_id for m in found.get(o.site.id, [])}
                added = 0
                for ms in footprints.in_polygons(extra_areas).values():
                    for m in ms:
                        if m.overture_id not in have and (include_carports or structure_kind(m) == "building"):
                            found[o.site.id].append(m)
                            have.add(m.overture_id)
                            added += 1
                o.reasons.append(f"added_{added}_buildings_from_{len(extra_areas)}_more_parcels_same_owner")
            if other_owner:
                o.reasons.append(f"skipped_{other_owner}_parcels_other_owner_under_unit_addresses")

    # Single-family rental communities (build-to-rent houses or townhomes on
    # their own lots): the address is one lot, the property is hundreds. Size
    # a sample of the nearest homes, checking each one's owner (one Regrid
    # record apiece), and scale kW per home to the unit count in the input.
    community: dict[str, list[int]] = {}  # site id -> homes in each found building
    if community_sample > 0:
        for o in outcomes:
            sample = _community_sample(o, found, footprints, parcels if community_owner_checks else None,
                                       unit_addresses, community_sample)
            if sample is not None:
                found[o.site.id], community[o.site.id] = sample

    # Otherwise: buildings under the county's per-unit address points.
    if unit_addresses is not None and campus_m == 0:
        unit_pts: dict[str, tuple[float, float]] = {}
        for o in outcomes:
            if not (o.geocode and found.get(o.site.id)) or o.site.id in parcel_sites:
                continue
            pts = site_units.get(o.site.id, [])
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
    progress(f"Sizing {len(jobs)} buildings (Google lookups first; uncached ones take a few seconds each)...")
    results = estimate_many([b for _o, b in jobs], geometric, google_client, calibrator, policy, workers, google_max_points,
                            equipment_client)
    for (o, _b), est in zip(jobs, results):
        o.estimates.append(est)
    for o in outcomes:
        o.counted = [True] * len(o.estimates)
        if o.site.id not in parcel_sites or len(o.estimates) < 2:
            continue
        with_google = [e.method == "google_filtered" for e in o.estimates]
        if any(with_google) and not all(with_google):
            # On a parcel Google otherwise covers, structures without Google data are
            # usually garages or sheds: reported, not counted. Unless they hold most of
            # the roof area (1.5x Google's), in which case Google missed the main buildings.
            area = lambda want: sum(_area_m2(b) for b, g in zip(o.buildings, with_google) if g == want)
            if area(False) > 1.5 * area(True):
                o.reasons.append(f"counted_{with_google.count(False)}_buildings_without_google_data_from_outlines")
            else:
                # Small ones are garages and sheds; a building-sized roof Google has no
                # data for (e.g. newer imagery gap) is counted from its outline and reviewed.
                big = [not g and _area_m2(b) >= big_building_m2 and b.structure == "building"
                       for b, g in zip(o.buildings, with_google)]
                o.counted = [g or k for g, k in zip(with_google, big)]
                dropped = with_google.count(False) - sum(big)
                if dropped:
                    o.reasons.append(f"not_counted_{dropped}_buildings_without_google_data")
                if any(big):
                    o.reasons.append(f"counted_{sum(big)}_large_buildings_without_google_data_from_outlines")
                    o.manual_review_extra.append(f"{sum(big)}_large_building_without_google_data_sized_from_outline")
        # Carports and garages count only when Google sized them (so shading is known)
        # and their typical panel yields >= carport_min_energy_ratio of the best roof panel.
        best = max((e.google.details.get("best_panel_kwh", 0.0) for b, e in zip(o.buildings, o.estimates)
                    if e.google and b.structure == "building"), default=0.0)
        shaded = no_data = small = 0
        for i, (b, e) in enumerate(zip(o.buildings, o.estimates)):
            if b.structure == "building" or not o.counted[i]:
                continue
            if e.method != "google_filtered":
                o.counted[i] = False
                no_data += 1
            elif best and e.google.details.get("median_panel_kwh", 0.0) < carport_min_energy_ratio * best:
                o.counted[i] = False
                shaded += 1
            elif e.dc_kw < carport_min_kw:  # Ivy: only where a sizeable system fits
                o.counted[i] = False
                small += 1
        if shaded:
            o.reasons.append(f"not_counted_{shaded}_shaded_carports")
        if small:
            o.reasons.append(f"not_counted_{small}_carports_or_garages_under_{carport_min_kw:g}_kw")
        if no_data and not any("without_google_data" in r for r in o.reasons):
            o.reasons.append(f"not_counted_{no_data}_carports_without_google_data")
    # Outline-only sizing ignores rooftop equipment. Scale it by the typical ratio of
    # Google-filtered to outline-only kW (median 0.57 over 45 test sites, Oct 2026).
    for o in outcomes:
        for b, e, c in o._triples():
            if c and e.method == "geometric" and e.primary and outline_only_factor != 1.0:
                e.dc_kw *= outline_only_factor
                e.reasons.append(f"outline_only_scaled_{outline_only_factor:g}")
    for o in outcomes:
        o.manual_review = imagery_concerns(o, stale_imagery_years, min_imagery_coverage, today or date.today())
        o.manual_review += [r for r in o.manual_review_extra if r not in o.manual_review]
        if not review_imagery_flags:
            # Reviewed by hand on 515 Greystar sites (Oct 2026), these were kept as
            # sized nearly every time: old imagery, outline-only roofs, sparse Google
            # panels. They stay in the reasons column, not the manual-review list.
            demoted = [r for r in o.manual_review if r.startswith(_DEMOTED_REVIEW)
                       or r.endswith("_large_building_without_google_data_sized_from_outline")]
            o.manual_review = [r for r in o.manual_review if r not in demoted]
            o.reasons += [r for r in demoted if r not in o.reasons]
        # Under ~0.1 kW per home: the buildings found can't be the whole property
        # (garden and mid-rise MaxFit designs run ~0.7-3 kW per unit).
        if (o.site.units and o.buildings and min_kw_per_unit > 0 and o.site.id not in community
                and o.dc_kw < min_kw_per_unit * o.site.units):
            o.manual_review.append(f"only_{o.dc_kw / o.site.units:.2f}_kw_per_unit_buildings_likely_missing")
        # One building holding far more than an apartment roof per home (or mapped as
        # a store, office or warehouse): the address probably landed on a neighbour.
        counted_b = [b for b, _e, c in o._triples() if c]
        classes = {(m.building_class or "") for m in o.matches}
        if o.site.units and counted_b and o.site.occupancy != Occupancy.COMMERCIAL:
            if len(counted_b) == 1 and o.dc_kw > max_kw_per_unit_one_building * o.site.units:
                o.manual_review.append(f"one_building_{o.dc_kw / o.site.units:.1f}_kw_per_unit_check_match")
            elif classes and classes <= _NONRESIDENTIAL_CLASSES:
                o.manual_review.append("matched_building_mapped_as_" + sorted(classes)[0] + "_check_match")
        _units_estimate(o, community, min_floor_m2_per_unit, kw_per_unit_fallback)
        if o.site.id in community:
            _scale_to_units(o, community[o.site.id], min_sample=min(5, community_sample))
    return outcomes


def _community_sample(o: SiteOutcome, found: dict, footprints, parcels, unit_addresses, n: int,
                      unit_m2: float = 100.0):
    """(buildings, homes per building) for a single-family community whose
    input unit count far exceeds the homes found, else None."""
    units = o.site.units
    if not o.geocode or not units or units < 10:
        return None
    # Only properties the input calls single-family / townhome / build-to-rent:
    # apartment and senior complexes with small buildings must not be scaled.
    if o.site.occupancy != Occupancy.R3:
        return None
    have = [m for m in found.get(o.site.id, []) if structure_kind(m) == "building"]
    if len(have) >= 5:  # the parcel already brought in the property's buildings
        return None
    zipcode = (_address_with_zip(o).rsplit(" ", 1)[-1:] or [""])[0]
    zipcode = zipcode if re.fullmatch(r"\d{5}", zipcode) else ""

    def homes(ms):
        counted = unit_addresses.homes_in(zipcode, [m.footprint for m in ms]) \
            if unit_addresses is not None and hasattr(unit_addresses, "homes_in") and zipcode else [0] * len(ms)
        # No address points on a building: a house, or a townhome row of ~unit_m2 per home.
        return [c or max(1, round(_fp_area_m2(m.footprint) / unit_m2)) if _fp_area_m2(m.footprint) >= 300 else c or 1
                for c, m in zip(counted, ms)]

    have_homes = homes(have) if have else []
    if sum(have_homes) * 2 >= units:
        return None
    radius = min(1000.0, max(150.0, 1.6 * math.sqrt(units * 450.0 / math.pi)))
    near = footprints.find({o.site.id: (o.geocode.lon, o.geocode.lat)}, search_m=radius, campus_m=radius).get(o.site.id, [])
    ids = {m.overture_id for m in found.get(o.site.id, [])}
    cands = sorted((m for m in near if m.overture_id not in ids and structure_kind(m) == "building"
                    and _fp_area_m2(m.footprint) <= 2000.0), key=lambda m: m.distance_m)
    owner = o.parcel if parcels is not None and o.parcel is not None else None
    picked, checked, other = [], 0, 0
    for m in cands:
        if len(picked) + len(have) >= n or checked >= 3 * n:
            break
        if owner is not None:
            checked += 1
            c = m.footprint.representative_point()
            try:
                p = parcels.parcel_at(c.y, c.x)
            except Exception:
                continue
            if not same_owner(p, owner):
                other += 1
                continue
        picked.append(m)
    out = found.get(o.site.id, []) + picked
    o.reasons.append(f"community_sampled_{len(have) + len(picked)}_buildings_within_{radius:.0f}m"
                     + (f"_skipped_{other}_other_owners" if other else "")
                     + ("" if owner is not None else "_ownership_not_checked"))
    is_home = [structure_kind(m) == "building" for m in out]
    counts = iter(homes([m for m, h in zip(out, is_home) if h]))
    return out, [next(counts) if h else 0 for h in is_home]


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def _scale_to_units(o: SiteOutcome, homes_per_building: list[int], min_sample: int,
                    max_kw_per_home: float = 20.0) -> None:
    """Scale the sampled community's rooftop kW to the input's unit count."""
    if len(homes_per_building) != len(o.estimates):
        return
    rows = [(e, h) for (b, e, c), h in zip(o._triples(), homes_per_building) if c and b.structure == "building" and h]
    homes, kw = sum(h for _e, h in rows), sum(e.dc_kw for e, _h in rows)
    if not homes or homes >= o.site.units:
        return
    if len(rows) < min_sample:  # too few to stand for the community: report what was found, unscaled
        o.manual_review.append(f"community_sample_only_{len(rows)}_buildings_not_scaled")
        return
    if kw / homes > max_kw_per_home:  # homes per building undercounted (e.g. one address for a fourplex)
        o.manual_review.append(f"community_{kw / homes:.0f}_kw_per_home_implausible_not_scaled")
        return
    per_home = sorted(e.dc_kw / h for e, h in rows)
    if per_home[-1] > 3 * max(per_home[len(per_home) // 4], 0.1):  # mixed sample (e.g. houses plus a clubhouse)
        o.manual_review.append(f"community_sample_varies_{per_home[0]:.0f}_to_{per_home[-1]:.0f}_kw_per_home")
    factor = o.site.units / homes
    for e, _h in rows:
        _scale_estimate(e, factor)
    o.reasons.append(f"community_{o.site.units}_homes_from_{homes}_sampled_{kw / homes:.1f}_kw_per_home")


def _scale_estimate(e: SiteEstimate, factor: float) -> None:
    e.dc_kw *= factor
    e.raw_kw *= factor
    for r in {id(r): r for r in (e.primary, e.geometric, e.google) if r is not None}.values():
        r.dc_kw *= factor
        for k, v in list(r.details.items()):
            if k.endswith("_kw") and isinstance(v, (int, float)) and not isinstance(v, bool):
                r.details[k] = v * factor


def _address_with_zip(o: SiteOutcome) -> str:
    """The site address, with a ZIP appended from the geocoder or parcel when missing."""
    addr = (o.site.address or "").strip()
    if not addr or re.search(r"\b\d{5}(?:-\d{4})?\s*(?:,?\s*(?:USA|US|United States))?\s*$", addr):
        return addr
    for source in ((o.geocode.matched_address if o.geocode else ""), getattr(o.parcel, "zip_code", "")):
        m = re.search(r"\b(\d{5})(?:-\d{4})?\b(?!.*\b\d{5}\b)", source or "")
        if m:
            return f"{addr} {m.group(1)}"
    return addr


_CA_ADDRESS = re.compile(r",\s*(CA|California)\b(\s+9\d{4})?", re.IGNORECASE)


def existing_solar_check(o: SiteOutcome, first_year: int = 2020) -> str:
    """Why a California site may already have solar, else "".

    California's energy code has required solar on new low-rise homes and apartments
    since 2020 and on all new multifamily since 2023. On the Oct 2026 Greystar review,
    every site omitted for existing solar was built 2023 or later or was too new for
    the map data; none built before 2020 had it.
    """
    text = " ".join(filter(None, (o.site.address, o.geocode.matched_address if o.geocode else "")))
    if not _CA_ADDRESS.search(text):
        return ""
    if o.site.year_built and o.site.year_built >= first_year:
        return f"built_{o.site.year_built}"
    if o.site.year_built is None and o.geocode and (
            not o.buildings or any(r.startswith(("google_sees_", "no_google_imagery")) for r in o.reasons + o.manual_review)):
        return "new_construction"
    return ""


_DEMOTED_REVIEW = ("google_imagery_", "most_roof_area_sized_from_outlines", "no_google_imagery_sized_from_outline",
                   "google_sees_")


def _units_estimate(o: SiteOutcome, community: dict, min_floor_m2: float, kw_per_unit: float,
                    storey_m: float = 3.3, default_storeys: int = 2) -> None:
    """Units x typical kW/unit when the buildings found can't hold the input's units.

    An apartment needs ~50-100 m2 of floor; buildings holding under min_floor_m2 per
    unit (footprint x storeys from mapped heights, 2 storeys where unmapped) are a
    fraction of the property: a leasing office, one phase, or new construction the
    maps don't show yet. On the Oct 2026 Greystar review, hand estimates for such
    sites ran a median 1.7 kW/unit, and nearly all were within 30% of units x 1.7.
    """
    units = o.site.units
    if not units or not o.geocode or o.site.id in community or o.site.occupancy == Occupancy.COMMERCIAL:
        return
    floor = 0.0
    for b, m in zip(o.buildings, o.matches or [None] * len(o.buildings)):
        if b.structure != "building":
            continue
        h = getattr(m, "height_m", None) if m is not None else None
        storeys = max(1, round(h / storey_m)) if h else default_storeys
        floor += _area_m2(b) * storeys
    o.floor_m2_per_unit = floor / units
    if min_floor_m2 > 0 and kw_per_unit > 0 and o.floor_m2_per_unit < min_floor_m2:
        o.units_estimate_kw = units * kw_per_unit
        if o.units_estimate_kw > o.dc_kw:
            reason = (f"buildings_found_hold_{o.floor_m2_per_unit:.0f}_m2_floor_per_unit_"
                      f"estimated_{kw_per_unit:g}_kw_per_unit")
            o.manual_review = [r for r in o.manual_review if not r.endswith("_kw_per_unit_buildings_likely_missing")]
            o.manual_review.append(reason)


def _area_m2(b: Building) -> float:
    return _fp_area_m2(b.footprint)


def _fp_area_m2(footprint) -> float:
    return LocalFrame.for_geometry(footprint).to_local(footprint).area


def imagery_concerns(o: SiteOutcome, stale_years: float, min_coverage: float, today: date) -> list[str]:
    """Reasons the site total can't be trusted without a look at current imagery.

    Kept narrow on purpose (it should catch a few percent of sites, not most):
    - no building found, an uncertain building match, or most of the roof sized
      from outlines only (no Google data);
    - Google panels cover under min_coverage of what the footprints hold: usually
      imagery taken before the building or its roof was finished;
    - the imagery behind most of the kW is older than stale_years.
    """
    if not o.buildings:
        return ["no_building_found"]
    out = []
    if any(r.startswith(("two_buildings_equally_close", "address_point_")) for r in o.reasons):
        out.append("building_match_uncertain")
    counted = [(b, e) for b, e, c in o._triples() if c]
    with_google = [(b, e) for b, e in counted if e.google]
    if not with_google:
        return out + ["no_google_imagery_sized_from_outline"]
    no_google_m2 = sum(_area_m2(b) for b, e in counted if not e.google)
    if no_google_m2 > 1.5 * sum(_area_m2(b) for b, _e in with_google):
        out.append("most_roof_area_sized_from_outlines")
    # Google's own panel count before any of our rules (setbacks, equipment,
    # shade): a crowded or setback-heavy roof is not a sign of stale imagery.
    goo = sum(e.google.details.get("google_unclipped_kw", e.google.dc_kw) for _b, e in with_google)
    geo = sum(e.geometric.dc_kw for _b, e in with_google if e.geometric)
    if geo > 0 and goo < min_coverage * geo:
        out.append(f"google_sees_{goo / geo:.0%}_of_footprint_capacity_imagery_may_predate_building")
    # Age of the imagery behind most of the kW, not the oldest shed on the parcel.
    by_date: dict[str, float] = {}
    for _b, e in with_google:
        d = e.google.details.get("imagery_date", "")
        if d and d[:4].isdigit() and int(d[:4]) > 0:
            by_date[d] = by_date.get(d, 0.0) + e.google.dc_kw
    if by_date:
        d = max(by_date, key=by_date.get)
        age = today.year + (today.month - 1) / 12 - (int(d[:4]) + (int(d[5:7]) - 1) / 12)
        if age > stale_years:
            out.append(f"google_imagery_{d}_{age:.0f}_years_old")
    return out
