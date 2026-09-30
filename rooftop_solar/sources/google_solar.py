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
from shapely.geometry import box

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

    @classmethod
    def from_response(cls, data: dict) -> "GoogleInsights":
        sp = data.get("solarPotential", {})
        d = data.get("imageryDate", {})
        return cls(
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
        kinds = set()
        for p in insights.panels:
            seg = insights.segments[p.segment_index] if p.segment_index < len(insights.segments) else GoogleSegment(0, 180)
            cos_p = math.cos(math.radians(seg.pitch_deg))
            # LANDSCAPE: long edge perpendicular to segment azimuth; PORTRAIT: parallel
            if p.orientation == "PORTRAIT":
                across, down = insights.panel_width_m, insights.panel_height_m
            else:
                across, down = insights.panel_height_m, insights.panel_width_m
            x, y = frame.point_to_local(p.lon, p.lat)
            rect = box(-across / 2, -down * cos_p / 2, across / 2, down * cos_p / 2)
            rect = affinity.translate(affinity.rotate(rect, -seg.azimuth_deg, origin=(0, 0)), x, y)
            if rect.intersection(zone).area < self.min_inside_fraction * rect.area:
                continue
            kept.append(rect)
            area = across * down
            kept_surface_m2 += area
            is_flat = seg.pitch_deg < flat_threshold
            kinds.add("flat" if is_flat else "pitched")
            if is_flat:
                flat_surface_m2 += area

        # Google lays panels flush on flat roofs; tilted racking needs row spacing.
        flat_density = self._flat_density(frame.lat0)
        effective_m2 = (kept_surface_m2 - flat_surface_m2) + flat_surface_m2 * flat_density
        count = int(effective_m2 // design.module.area_m2)

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
                "google_panels_kept": len(kept),
                "google_max_array_panels": insights.max_array_panels,
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
