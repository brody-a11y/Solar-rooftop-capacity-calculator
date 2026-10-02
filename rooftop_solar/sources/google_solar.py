"""Google Solar API (buildingInsights) client and a code-filtered estimator.

Google's panel layout comes from its DSM, so it sees HVAC units, vents and other
roof clutter that a footprint cannot. It does not apply fire-code pathways. The
estimator below keeps only Google panels inside the code-compliant zone and
rescales to the design module.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import requests
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

from ..geometry import LocalFrame, polygons
from ..models import FT, Building, Racking, SizingResult
from ..sizing import GeometricEstimator

API_URL = "https://solar.googleapis.com/v1/buildingInsights:findClosest"


@dataclass
class GoogleSegment:
    pitch_deg: float
    azimuth_deg: float


@dataclass
class GooglePanel:
    lon: float
    lat: float
    orientation: str  # LANDSCAPE or PORTRAIT
    segment_index: int
    yearly_kwh: float = 0.0  # Google's modelled yearly DC energy for this panel


@dataclass
class GoogleInsights:
    imagery_quality: str
    imagery_date: str
    panel_height_m: float  # long side
    panel_width_m: float
    panel_watts: float
    max_array_panels: int
    segments: list[GoogleSegment]
    panels: list[GooglePanel]
    name: str = ""
    center: tuple[float, float] | None = None  # (lon, lat)
    buildings_merged: int = 1

    @classmethod
    def from_response(cls, data: dict) -> "GoogleInsights":
        sp = data.get("solarPotential", {})
        d = data.get("imageryDate", {})
        c = data.get("center") or {}
        return cls(
            name=data.get("name", ""),
            center=(c["longitude"], c["latitude"]) if "longitude" in c else None,
            imagery_quality=data.get("imageryQuality", "UNKNOWN"),
            imagery_date=f"{d.get('year', 0):04d}-{d.get('month', 0):02d}-{d.get('day', 0):02d}" if d else "",
            panel_height_m=sp.get("panelHeightMeters", 0.0),
            panel_width_m=sp.get("panelWidthMeters", 0.0),
            panel_watts=sp.get("panelCapacityWatts", 0.0),
            max_array_panels=sp.get("maxArrayPanelsCount", 0),
            segments=[
                GoogleSegment(s.get("pitchDegrees", 0.0), s.get("azimuthDegrees", 180.0))
                for s in sp.get("roofSegmentStats", [])
            ],
            panels=[
                GooglePanel(
                    p["center"]["longitude"],
                    p["center"]["latitude"],
                    p.get("orientation", "LANDSCAPE"),
                    p.get("segmentIndex", 0),
                    float(p.get("yearlyEnergyDcKwh", 0.0) or 0.0),
                )
                for p in sp.get("solarPanels", [])
            ],
        )


def _segment(insights: GoogleInsights, p: GooglePanel) -> GoogleSegment:
    return insights.segments[p.segment_index] if p.segment_index < len(insights.segments) else GoogleSegment(0, 180)


def _panel_dims(insights: GoogleInsights, p: GooglePanel) -> tuple[float, float]:
    """(across-slope, down-slope) panel size. LANDSCAPE: long edge perpendicular
    to the segment azimuth; PORTRAIT: parallel."""
    if p.orientation == "PORTRAIT":
        return insights.panel_width_m, insights.panel_height_m
    return insights.panel_height_m, insights.panel_width_m


def _panel_rect(insights: GoogleInsights, p: GooglePanel, seg: GoogleSegment, frame: LocalFrame):
    """Plan-view panel rectangle in `frame` coordinates."""
    across, down = _panel_dims(insights, p)
    cos_p = math.cos(math.radians(seg.pitch_deg))
    x, y = frame.point_to_local(p.lon, p.lat)
    rect = box(-across / 2, -down * cos_p / 2, across / 2, down * cos_p / 2)
    return affinity.translate(affinity.rotate(rect, -seg.azimuth_deg, origin=(0, 0)), x, y)


def footprint_from_insights(insights: GoogleInsights, pad_m: float = 1.6, close_m: float = 3.0):
    """Approximate roof outline (lon/lat) from Google's panel layout, for sites the
    footprint dataset doesn't have yet (new construction). Panels are merged,
    gaps up to 2 x `close_m` are closed, and the outline is padded by `pad_m` so
    the fire-code perimeter lands roughly on Google's own edge margin."""
    if not insights.panels:
        return None
    lon0, lat0 = insights.center or (insights.panels[0].lon, insights.panels[0].lat)
    frame = LocalFrame(lon0, lat0)
    rects = unary_union([_panel_rect(insights, p, _segment(insights, p), frame) for p in insights.panels])
    outline = rects.buffer(close_m, join_style="mitre").buffer(pad_m - close_m, join_style="mitre")
    parts = polygons(outline)
    if not parts:
        return None
    return frame.to_lonlat(max(parts, key=lambda g: g.area))


_QUALITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def merge_insights(parts: list[GoogleInsights]) -> GoogleInsights:
    """Combine several Google 'buildings' into one. Google often splits a large
    podium building or a complex into separate entries."""
    first = parts[0]
    segments, panels = [], []
    for part in parts:
        offset = len(segments)
        segments.extend(part.segments)
        panels.extend(GooglePanel(p.lon, p.lat, p.orientation, p.segment_index + offset, p.yearly_kwh) for p in part.panels)
    worst = min(parts, key=lambda x: _QUALITY_RANK.get(x.imagery_quality, 0))
    return GoogleInsights(
        imagery_quality=worst.imagery_quality,
        imagery_date=min(p.imagery_date for p in parts),
        panel_height_m=first.panel_height_m,
        panel_width_m=first.panel_width_m,
        panel_watts=first.panel_watts,
        max_array_panels=sum(p.max_array_panels for p in parts),
        segments=segments,
        panels=panels,
        name="+".join(p.name for p in parts),
        center=first.center,
        buildings_merged=len(parts),
    )


def _share_on_footprint(ins: GoogleInsights, frame: LocalFrame, zone) -> float:
    """Fraction of a Google building's panels (or its centre, if it has none)
    that fall on our footprint."""
    if ins.panels:
        inside = sum(zone.contains(Point(*frame.point_to_local(p.lon, p.lat))) for p in ins.panels)
        return inside / len(ins.panels)
    if ins.center is not None:
        return float(zone.contains(Point(*frame.point_to_local(*ins.center))))
    return 0.0


def _candidate_points(local, spacing_m: float = 8.0) -> list[tuple[float, float]]:
    """Interior grid points of a local-frame footprint, representative point first."""
    rp = local.representative_point()
    inner = local.buffer(-2.0)
    if inner.is_empty:
        inner = local
    minx, miny, maxx, maxy = local.bounds
    pts = [(rp.x, rp.y)]
    y = miny + spacing_m / 2
    while y < maxy:
        x = minx + spacing_m / 2
        while x < maxx:
            if inner.contains(Point(x, y)):
                pts.append((x, y))
            x += spacing_m
        y += spacing_m
    return pts


def fetch_building(client: "GoogleSolarClient", footprint, max_points: int = 9, match_buffer_m: float = 10.0,
                   min_share: float = 0.5, cover_m: float = 6.0) -> GoogleInsights:
    """Query Google over a footprint and merge the distinct buildings that mostly
    (>= `min_share` of their panels) sit on it.

    Google often splits one large or L-shaped building into several entries, so
    lookups are adaptive: after each one, the area covered by the returned
    building's panels is marked done, and the next lookup goes to the uncovered
    point farthest from earlier lookups. At most `max_points` lookups are made.
    If every MEDIUM-quality lookup returns 404, one LOW-quality retry is made.
    """
    frame = LocalFrame.for_geometry(footprint)
    local = frame.to_local(footprint)
    zone = local.buffer(match_buffer_m)
    candidates = _candidate_points(local)
    errors: list[requests.HTTPError] = []
    for quality in ("MEDIUM", "LOW"):
        found: dict[str, GoogleInsights] = {}
        errors = []
        covered = Polygon()
        chosen: list[tuple[float, float]] = []
        budget = max_points if quality == "MEDIUM" else 1
        while len(chosen) < budget:
            open_ = [c for c in candidates if not covered.contains(Point(c))]
            if not open_:
                break
            pick = open_[0] if not chosen else max(
                open_, key=lambda c: min(math.hypot(c[0] - q[0], c[1] - q[1]) for q in chosen))
            chosen.append(pick)
            lon, lat = frame.point_to_lonlat(*pick)
            try:
                ins = GoogleInsights.from_response(client.building_insights(lat, lon, required_quality=quality))
            except requests.HTTPError as exc:
                errors.append(exc)
                covered = covered.union(Point(pick).buffer(cover_m * 2))
                continue
            done = Point(pick).buffer(cover_m * 2)
            if ins.panels:
                done = done.union(unary_union(
                    [_panel_rect(ins, p, _segment(ins, p), frame) for p in ins.panels]).buffer(cover_m))
            covered = covered.union(done)
            if ins.name not in found and _share_on_footprint(ins, frame, zone) >= min_share:
                found[ins.name] = ins
        if found:
            return merge_insights(list(found.values()))
        if not errors or any(_status(e) != 404 for e in errors):
            break
    if errors:
        raise GoogleLookupError(f"http_{_status(errors[0])}") from errors[0]
    raise GoogleLookupError("no_google_building_on_footprint")


def _status(exc: requests.HTTPError) -> int | None:
    return exc.response.status_code if exc.response is not None else None


class GoogleLookupError(RuntimeError):
    """Google returned nothing usable for a footprint; str() is a short reason code."""


class GoogleSolarClient:
    """Thin client with an on-disk cache. Every uncached call is billed by Google."""

    def __init__(self, api_key: str, cache_dir: str | Path | None = ".cache/google_solar", timeout: float = 30.0):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.session = requests.Session()

    def building_insights(self, lat: float, lon: float, required_quality: str = "MEDIUM") -> dict:
        key = hashlib.sha1(f"{lat:.7f},{lon:.7f},{required_quality}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            data = json.loads(path.read_text())
            if "_http_status" in data:  # a saved 404: no building at this point
                resp = requests.Response()
                resp.status_code = data["_http_status"]
                raise requests.HTTPError(f"{resp.status_code} (cached)", response=resp)
            return data
        params = {
            "location.latitude": f"{lat:.7f}",
            "location.longitude": f"{lon:.7f}",
            "requiredQuality": required_quality,
            "key": self.api_key,
        }
        for attempt in range(6):  # back off when Google rate-limits (429) or errors (5xx)
            resp = self.session.get(API_URL, params=params, timeout=self.timeout)
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 5:
                break
            time.sleep(min(60.0, 2.0 * 2 ** attempt))
        if resp.status_code == 404 and path:
            # Google has no building here; remember that so reruns don't ask again.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"_http_status": 404}))
        resp.raise_for_status()
        data = resp.json()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return data


DETECTED_KINDS = ("equipment", "small_equipment")  # from Google's surface model (google_dsm)


class GoogleFilteredEstimator:
    method = "google_filtered"

    def __init__(self, geometric: GeometricEstimator | None = None, min_inside_fraction: float = 0.95):
        self.geometric = geometric or GeometricEstimator()
        self.min_inside_fraction = min_inside_fraction

    def estimate(self, building: Building, insights: GoogleInsights) -> SizingResult:
        design = self.geometric.design
        frame = LocalFrame.for_geometry(building.footprint)
        fp = frame.to_local(building.footprint)
        zone = self.geometric.usable_zone(building, frame)
        # Pitched multifamily roofs under the R-3 rules (IFC 1205.3 exception): the
        # plane-edge setback below replaces the commercial perimeter pathway, so
        # pitched panels only have to sit on the roof and clear of obstructions.
        residential_pitched = self.geometric.rules.uses_residential_rules(building.occupancy, pitched=True)
        pitched_zone = (fp.difference(self.geometric._keep_out(building.obstructions, frame, frame.lat0))
                        if residential_pitched else zone)
        equipment = [o for o in building.obstructions if o.kind in DETECTED_KINDS]
        # Raised racking spans detected equipment, so its zone ignores them.
        raised_zone = (self.geometric.usable_zone(replace(building, obstructions=[o for o in building.obstructions
                                                                                   if o.kind not in DETECTED_KINDS]), frame)
                       if equipment else zone)
        equipment_rects = []  # Google flat panels removed only for equipment clearance
        flat_threshold = design.flat_pitch_threshold_deg
        equipment_m2 = 0.0  # flat panels Google placed in equipment keep-out areas

        kept, kept_surface_m2, flat_surface_m2 = [], 0.0, 0.0
        all_surface_m2 = all_flat_m2 = 0.0  # every Google panel, ignoring the code zone
        poleward_m2 = 0.0  # pitched panels facing away from the sun (north in the US)
        low_yield_m2 = 0.0  # panels producing well below the building's best ones
        kept_kwh: list[float] = []
        kept_flat = []  # plan-view rectangles of kept panels on flat segments
        dropped = []  # panels left out for shade, north faces or pitch: raised racking can't fix those
        # Reference = 90th-percentile panel, not the single best: a few unusually
        # sunny edge panels on a big flat roof would otherwise set the bar so high
        # that most of a usable roof is dropped.
        energies = sorted(p.yearly_kwh for p in insights.panels)
        best_kwh = energies[int(0.9 * (len(energies) - 1))] if energies else 0.0
        min_kwh = design.min_panel_energy_ratio * best_kwh
        kinds = set()
        rects = [_panel_rect(insights, p, _segment(insights, p), frame) for p in insights.panels]
        pitched_ok = self._pitched_setback_zones(insights, rects, flat_threshold)
        setback_m2 = 0.0  # pitched panels inside the plane-edge setback
        for p, rect in zip(insights.panels, rects):
            seg = _segment(insights, p)
            across, down = _panel_dims(insights, p)
            is_flat = seg.pitch_deg < flat_threshold
            all_surface_m2 += across * down
            if is_flat:
                all_flat_m2 += across * down
            if not is_flat:
                dropped.append(rect)
                ok = pitched_ok.get(p.segment_index)
                if ok is not None and rect.intersection(ok).area < self.min_inside_fraction * rect.area:
                    setback_m2 += across * down
                    continue
            if rect.intersection(zone if is_flat else pitched_zone).area < self.min_inside_fraction * rect.area:
                if equipment and is_flat and rect.intersection(raised_zone).area >= self.min_inside_fraction * rect.area:
                    equipment_m2 += across * down
                    equipment_rects.append(rect)
                continue
            if not is_flat and self._poleward(seg.azimuth_deg, frame.lat0):
                poleward_m2 += across * down
                if design.exclude_poleward_faces:
                    continue
            if min_kwh > 0 and p.yearly_kwh < min_kwh:
                low_yield_m2 += across * down
                dropped.append(rect)
                continue
            kept.append(rect)
            kept_kwh.append(p.yearly_kwh)
            if is_flat:
                kept_flat.append(rect)
            area = across * down
            kept_surface_m2 += area
            kinds.add("flat" if is_flat else "pitched")
            if is_flat:
                flat_surface_m2 += area

        # Google lays panels flush on flat roofs; tilted racking needs row spacing.
        flat_density = self._flat_density(frame.lat0)
        effective_m2 = (kept_surface_m2 - flat_surface_m2) + flat_surface_m2 * flat_density
        count = int(effective_m2 // design.module.area_m2)
        # Google's own maximum for the building it found, without fire-code
        # pathways and not clipped to our footprint (which may be the wrong building).
        unclipped_m2 = (all_surface_m2 - all_flat_m2) + all_flat_m2 * flat_density
        unclipped_kw = int(unclipped_m2 // design.module.area_m2) * design.module.watts_dc / 1000.0
        # Raised racking: enclosed equipment gaps in the layout, plus the panels
        # Google placed that only the equipment clearance removed (bounded by
        # Google's own layout, so false detections can't inflate it).
        try:
            gap_modules, raised_gaps = self._raised_racking_modules(kept_flat, dropped + equipment_rects, raised_zone,
                                                                    flat_density, design)
        except GEOSException:  # invalid geometry from Google's panels: no gap estimate for this roof
            gap_modules, raised_gaps = 0, []
        equipment_modules = int(equipment_m2 * flat_density // design.module.area_m2)
        raised_extra = gap_modules + equipment_modules
        raised_gaps = raised_gaps + equipment_rects

        flags = []
        if 0 < count < design.min_modules_per_structure:
            flags.append(f"under_{design.min_modules_per_structure}_modules_not_designed")
            count, kept = 0, []
        if insights.imagery_quality not in ("HIGH", "MEDIUM"):
            flags.append(f"imagery_quality_{insights.imagery_quality.lower()}")
        if insights.panels and len(kept) == 0:
            flags.append("no_google_panels_in_code_zone_check_footprint_match")
        if not insights.panels:
            flags.append("google_returned_no_panels")

        return SizingResult(
            building_id=building.id,
            method=self.method,
            dc_kw=count * design.module.watts_dc / 1000.0,
            module_count=count,
            gross_roof_area_m2=fp.area,
            usable_area_m2=zone.area,
            roof_type="mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "flat"),
            flags=flags,
            details={
                "google_panels_total": len(insights.panels),
                "google_unclipped_kw": unclipped_kw,
                "raised_racking_extra_modules": raised_extra,
                "google_panels_kept": len(kept),
                "google_max_array_panels": insights.max_array_panels,
                "google_buildings_merged": insights.buildings_merged,
                "poleward_face_kw": round(int(poleward_m2 // design.module.area_m2) * design.module.watts_dc / 1000.0, 1),
                "poleward_faces_excluded": design.exclude_poleward_faces,
                "best_panel_kwh": max(kept_kwh, default=0.0),
                "median_panel_kwh": sorted(kept_kwh)[len(kept_kwh) // 2] if kept_kwh else 0.0,
                "equipment_detected": len(equipment),
                "equipment_clearance_kw": round(int(equipment_m2 * flat_density // design.module.area_m2) * design.module.watts_dc / 1000.0, 1),
                "pitched_setback_kw": round(int(setback_m2 // design.module.area_m2) * design.module.watts_dc / 1000.0, 1),
                "low_yield_kw": round(int(low_yield_m2 // design.module.area_m2) * design.module.watts_dc / 1000.0, 1),
                "imagery_quality": insights.imagery_quality,
                "imagery_date": insights.imagery_date,
                "flat_density_factor": round(flat_density, 3),
            },
            layout=[frame.to_lonlat(r) for r in kept],
            footprint=building.footprint,
            raised_areas=[frame.to_lonlat(g) for g in raised_gaps],
            equipment=[o.geometry for o in equipment],
        )

    @staticmethod
    def _raised_racking_modules(flat_rects, dropped_rects, zone, flat_density: float, design) -> tuple[int, list]:
        """Extra modules if raised racking spans equipment gaps on flat roofs.

        A gap counts when it is (a) inside the code-compliant zone, (b) narrow
        enough to span (no wider than 2 x raised_gap_m), (c) at least ~1.5 m
        wide, so the ragged ends of panel rows don't count, and (d) mostly
        surrounded by panels: at least 60% of its edge borders the layout.
        That takes enclosed holes and equipment notches open to one side, and
        leaves out strips along the fire setback, courtyards and wells.
        Spots where Google's panels were dropped (shaded, north-facing or
        pitched) are not gaps: raising the racking doesn't make them usable.
        An estimate: equipment heights and structural spans aren't checked.
        """
        if not flat_rects or design.raised_gap_m <= 0:
            return 0, []
        g = design.raised_gap_m
        layout = unary_union(flat_rects).buffer(0.5, join_style="mitre").buffer(-0.5, join_style="mitre")
        # opening drops slivers under ~1.5 m wide: setback edges, ragged row ends
        layout = make_valid(layout)
        zone = make_valid(zone)
        blocked = unary_union([layout, *dropped_rects]) if dropped_rects else layout
        free = zone.difference(blocked).buffer(-0.75, join_style="mitre").buffer(0.75, join_style="mitre")
        near_layout = layout.buffer(0.3, join_style="mitre")
        gaps = []
        for gap in polygons(make_valid(free)):
            if gap.length == 0:
                continue
            try:
                if gap.buffer(-g).is_empty and gap.boundary.intersection(near_layout).length >= 0.6 * gap.length:
                    gaps.append(gap)
            except GEOSException:  # a degenerate sliver; skip it rather than fail the building
                continue
        return int(sum(gap.area for gap in gaps) * flat_density // design.module.area_m2), gaps

    def _pitched_setback_zones(self, insights: GoogleInsights, rects, flat_threshold: float) -> dict:
        """Per pitched roof segment, the area at least `pitched_setback` inside the
        edge of Google's array on that plane.

        Google doesn't return roof-plane outlines, and on pitched roofs its
        layouts run close to the plane edges (ridge, eaves, hips, rakes). The
        array's own outline stands in for the plane, so panels within the setback
        of it are dropped (IFC 1205.2: 36 in, 18 in where the AHJ allows). An
        approximation: where Google already left a margin this over-trims.
        """
        setback = self.geometric.rules.residential_setback_ft * FT
        if setback <= 0:
            return {}
        by_seg: dict[int, list] = {}
        for p, rect in zip(insights.panels, rects):
            if _segment(insights, p).pitch_deg >= flat_threshold:
                by_seg.setdefault(p.segment_index, []).append(rect)
        zones = {}
        for idx, rs in by_seg.items():
            try:
                array = unary_union(rs).buffer(0.3, join_style="mitre").buffer(-0.3, join_style="mitre")
                zones[idx] = array.buffer(-setback, join_style="mitre")
            except GEOSException:
                continue
        return zones

    def _poleward(self, azimuth_deg: float, lat: float) -> bool:
        pole = 0.0 if lat >= 0 else 180.0
        diff = abs((azimuth_deg - pole + 180.0) % 360.0 - 180.0)
        return diff < self.geometric.design.poleward_cone_deg

    def _flat_density(self, lat: float) -> float:
        """Module area per unit roof area for the configured flat racking (its GCR)."""
        d = self.geometric.design
        if d.flat_racking == Racking.FLUSH:
            return 1.0
        specs = self.geometric._flat_specs(lat)
        _orient, spec = specs[0]
        slant = spec.depth_m / math.cos(math.radians(d.flat_tilt_deg))
        return slant / spec.row_pitch_m
