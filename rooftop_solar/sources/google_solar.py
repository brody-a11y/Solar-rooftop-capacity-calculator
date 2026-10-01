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
from dataclasses import dataclass
from pathlib import Path

import requests
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from ..geometry import LocalFrame, polygons
from ..models import Building, Racking, SizingResult
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
        resp = self.session.get(
            API_URL,
            params={
                "location.latitude": f"{lat:.7f}",
                "location.longitude": f"{lon:.7f}",
                "requiredQuality": required_quality,
                "key": self.api_key,
            },
            timeout=self.timeout,
        )
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
        flat_threshold = design.flat_pitch_threshold_deg

        kept, kept_surface_m2, flat_surface_m2 = [], 0.0, 0.0
        all_surface_m2 = all_flat_m2 = 0.0  # every Google panel, ignoring the code zone
        poleward_m2 = 0.0  # pitched panels facing away from the sun (north in the US)
        low_yield_m2 = 0.0  # panels producing well below the building's best ones
        kept_kwh: list[float] = []
        best_kwh = max((p.yearly_kwh for p in insights.panels), default=0.0)
        min_kwh = design.min_panel_energy_ratio * best_kwh
        kinds = set()
        for p in insights.panels:
            seg = _segment(insights, p)
            across, down = _panel_dims(insights, p)
            is_flat = seg.pitch_deg < flat_threshold
            all_surface_m2 += across * down
            if is_flat:
                all_flat_m2 += across * down
            rect = _panel_rect(insights, p, seg, frame)
            if rect.intersection(zone).area < self.min_inside_fraction * rect.area:
                continue
            if not is_flat and self._poleward(seg.azimuth_deg, frame.lat0):
                poleward_m2 += across * down
                if design.exclude_poleward_faces:
                    continue
            if min_kwh > 0 and p.yearly_kwh < min_kwh:
                low_yield_m2 += across * down
                continue
            kept.append(rect)
            kept_kwh.append(p.yearly_kwh)
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
                "google_panels_kept": len(kept),
                "google_max_array_panels": insights.max_array_panels,
                "google_buildings_merged": insights.buildings_merged,
                "poleward_face_kw": round(int(poleward_m2 // design.module.area_m2) * design.module.watts_dc / 1000.0, 1),
                "poleward_faces_excluded": design.exclude_poleward_faces,
                "best_panel_kwh": max(kept_kwh, default=0.0),
                "median_panel_kwh": sorted(kept_kwh)[len(kept_kwh) // 2] if kept_kwh else 0.0,
                "low_yield_kw": round(int(low_yield_m2 // design.module.area_m2) * design.module.watts_dc / 1000.0, 1),
                "imagery_quality": insights.imagery_quality,
                "imagery_date": insights.imagery_date,
                "flat_density_factor": round(flat_density, 3),
            },
            layout=[frame.to_lonlat(r) for r in kept],
            footprint=building.footprint,
        )

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
