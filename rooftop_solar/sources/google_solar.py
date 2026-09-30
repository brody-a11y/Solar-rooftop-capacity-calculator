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
from shapely.geometry import Point, box

from ..geometry import LocalFrame
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
            imagery_date=f"{d.get('year', '')}-{d.get('month', '')}-{d.get('day', '')}" if d else "",
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
                )
                for p in sp.get("solarPanels", [])
            ],
        )


_QUALITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def merge_insights(parts: list[GoogleInsights]) -> GoogleInsights:
    """Combine several Google 'buildings' into one. Google often splits a large
    podium building or a complex into separate entries."""
    first = parts[0]
    segments, panels = [], []
    for part in parts:
        offset = len(segments)
        segments.extend(part.segments)
        panels.extend(GooglePanel(p.lon, p.lat, p.orientation, p.segment_index + offset) for p in part.panels)
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


def sample_points(footprint, spacing_m: float = 35.0, max_points: int = 9) -> list[tuple[float, float]]:
    """(lon, lat) query points spread over a footprint: its representative point,
    then a grid of interior points, thinned evenly to at most `max_points`."""
    frame = LocalFrame.for_geometry(footprint)
    local = frame.to_local(footprint)
    rp = local.representative_point()
    pts = [(rp.x, rp.y)]
    minx, miny, maxx, maxy = local.bounds
    grid = []
    y = miny + spacing_m / 2
    while y < maxy:
        x = minx + spacing_m / 2
        while x < maxx:
            if local.contains(Point(x, y)) and math.hypot(x - rp.x, y - rp.y) > spacing_m / 2:
                grid.append((x, y))
            x += spacing_m
        y += spacing_m
    room = max_points - 1
    if len(grid) > room > 0:
        step = len(grid) / room
        grid = [grid[int(i * step)] for i in range(room)]
    pts.extend(grid[: max(room, 0)])
    return [frame.point_to_lonlat(x, y) for x, y in pts]


def fetch_building(client: "GoogleSolarClient", footprint, max_points: int = 9,
                   spacing_m: float = 35.0, match_buffer_m: float = 10.0) -> GoogleInsights:
    """Query Google at several points over a footprint and merge the distinct
    buildings whose centres fall on (or within `match_buffer_m` of) it."""
    frame = LocalFrame.for_geometry(footprint)
    zone = frame.to_local(footprint).buffer(match_buffer_m)
    found: dict[str, GoogleInsights] = {}
    errors = []
    for lon, lat in sample_points(footprint, spacing_m, max_points):
        try:
            ins = GoogleInsights.from_response(client.building_insights(lat, lon))
        except requests.HTTPError as exc:
            errors.append(exc)
            continue
        if ins.name in found:
            continue
        if ins.center is not None and not zone.contains(Point(*frame.point_to_local(*ins.center))):
            continue  # Google's closest building is a neighbour
        found[ins.name] = ins
    if not found:
        if errors:
            raise errors[0]
        raise LookupError("Google found no building on this footprint")
    return merge_insights(list(found.values()))


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
            return json.loads(path.read_text())
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
        kinds = set()
        for p in insights.panels:
            seg = insights.segments[p.segment_index] if p.segment_index < len(insights.segments) else GoogleSegment(0, 180)
            cos_p = math.cos(math.radians(seg.pitch_deg))
            # LANDSCAPE: long edge perpendicular to segment azimuth; PORTRAIT: parallel
            if p.orientation == "PORTRAIT":
                across, down = insights.panel_width_m, insights.panel_height_m
            else:
                across, down = insights.panel_height_m, insights.panel_width_m
            is_flat = seg.pitch_deg < flat_threshold
            all_surface_m2 += across * down
            if is_flat:
                all_flat_m2 += across * down
            x, y = frame.point_to_local(p.lon, p.lat)
            rect = box(-across / 2, -down * cos_p / 2, across / 2, down * cos_p / 2)
            rect = affinity.translate(affinity.rotate(rect, -seg.azimuth_deg, origin=(0, 0)), x, y)
            if rect.intersection(zone).area < self.min_inside_fraction * rect.area:
                continue
            kept.append(rect)
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
                "imagery_quality": insights.imagery_quality,
                "imagery_date": insights.imagery_date,
                "flat_density_factor": round(flat_density, 3),
            },
            layout=[frame.to_lonlat(r) for r in kept],
            footprint=building.footprint,
        )

    def _flat_density(self, lat: float) -> float:
        """Module area per unit roof area for the configured flat racking (its GCR)."""
        d = self.geometric.design
        if d.flat_racking == Racking.FLUSH:
            return 1.0
        specs = self.geometric._flat_specs(lat)
        _orient, spec = specs[0]
        slant = spec.depth_m / math.cos(math.radians(d.flat_tilt_deg))
        return slant / spec.row_pitch_m
