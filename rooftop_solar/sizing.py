"""Geometric max-size estimator: footprint/roof planes + fire code -> module layout."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .fire_code import FireCodeRules
from .geometry import LocalFrame, inset, principal_axes, rotate, section_gaps
from .layout import RowSpec, pack_rows, south_row_pitch_m
from .models import FT, Building, Module, Obstruction, Racking, RoofPlane, SizingResult


@dataclass(frozen=True)
class DesignConfig:
    """Design assumptions. Racking values are placeholders: set them from the
    datasheet of the racking you actually specify, then calibrate."""

    module: Module = field(default_factory=Module)
    flat_racking: Racking = Racking.EAST_WEST
    flat_tilt_deg: float = 10.0
    east_west_gcr: float = 0.90
    south_gcr: float | None = None  # None: derive row pitch from winter-solstice shading
    flat_orientations: tuple[str, ...] = ("landscape",)
    pitched_orientations: tuple[str, ...] = ("portrait", "landscape")
    flat_pitch_threshold_deg: float = 9.46  # 2:12
    module_gap_m: float = 0.02
    edge_setback_ft: float = 0.0  # wind/structural edge zone; perimeter = max(fire, this)
    parapet_setback_ratio: float = 0.0  # extra setback = ratio * parapet height
    obstruction_shade_ratio: float = 0.0  # extend obstruction keep-out poleward by ratio * height
    phase_steps: int = 8


def _module_dims(module: Module, orientation: str) -> tuple[float, float]:
    """(along row, across row) slant dimensions."""
    if orientation == "landscape":
        return module.length_m, module.width_m
    return module.width_m, module.length_m


class GeometricEstimator:
    method = "geometric"

    def __init__(self, rules: FireCodeRules | None = None, design: DesignConfig | None = None):
        self.rules = rules or FireCodeRules()
        self.design = design or DesignConfig()

    # -- public ---------------------------------------------------------------

    def estimate(self, building: Building) -> SizingResult:
        frame = LocalFrame.for_geometry(building.footprint)
        fp = frame.to_local(building.footprint)
        lat = frame.lat0
        d = self.design

        perimeter = max(
            self.rules.perimeter_m(fp),
            d.edge_setback_ft * FT,
            d.parapet_setback_ratio * building.parapet_height_m,
        )
        commercial_base = inset(fp, perimeter)
        keep_out = self._keep_out(building.obstructions, frame, lat)

        planes = building.roof_planes or [RoofPlane(building.footprint, 0.0, 180.0)]
        flags: list[str] = []
        if not building.roof_planes:
            flags.append("no_roof_planes_assumed_flat")
        if not building.obstructions_mapped:
            flags.append("obstructions_not_mapped_upper_bound")

        modules: list[Polygon] = []
        usable_area = 0.0
        plane_details = []
        kinds = set()
        for i, plane in enumerate(planes):
            pm = frame.to_local(plane.geometry).intersection(fp)
            pitched = plane.pitch_deg >= d.flat_pitch_threshold_deg
            kinds.add("pitched" if pitched else "flat")
            if pitched:
                if self.rules.uses_residential_rules(building.occupancy, pitched=True):
                    usable = inset(pm, self.rules.residential_setback_ft * FT)
                    if "pitched_setbacks_approximated" not in flags:
                        flags.append("pitched_setbacks_approximated")
                else:
                    usable = pm.intersection(commercial_base)
                usable = usable.difference(keep_out)
                placed, info = self._pack_pitched(usable, plane)
            else:
                usable = pm.intersection(commercial_base).difference(keep_out)
                placed, info = self._pack_flat(usable, lat)
            usable_area += usable.area
            modules.extend(placed)
            plane_details.append({"plane": i, "pitch_deg": plane.pitch_deg, "modules": len(placed), **info})

        roof_type = "mixed" if len(kinds) > 1 else kinds.pop()
        count = len(modules)
        return SizingResult(
            building_id=building.id,
            method=self.method,
            dc_kw=count * d.module.watts_dc / 1000.0,
            module_count=count,
            gross_roof_area_m2=fp.area,
            usable_area_m2=usable_area,
            roof_type=roof_type,
            flags=flags,
            details={"perimeter_m": round(perimeter, 3), "planes": plane_details},
            layout=[frame.to_lonlat(m) for m in modules],
            footprint=building.footprint,
        )

    def usable_zone(self, building: Building, frame: LocalFrame) -> BaseGeometry:
        """Code-compliant zone in `frame` coordinates, used to filter third-party layouts."""
        fp = frame.to_local(building.footprint)
        d = self.design
        perimeter = max(
            self.rules.perimeter_m(fp), d.edge_setback_ft * FT, d.parapet_setback_ratio * building.parapet_height_m
        )
        zone = inset(fp, perimeter)
        _short, _long, angle = principal_axes(fp)
        aligned = rotate(zone, -angle)
        gaps = section_gaps(aligned, self.rules.max_array_section_ft * FT, self.rules.section_gap_ft * FT)
        zone = rotate(aligned.difference(gaps), angle)
        return zone.difference(self._keep_out(building.obstructions, frame, frame.lat0))

    # -- internals ------------------------------------------------------------

    def _keep_out(self, obstructions: list[Obstruction], frame: LocalFrame, lat: float) -> BaseGeometry:
        zones = []
        poleward = 1.0 if lat >= 0 else -1.0
        for ob in obstructions:
            g = frame.to_local(ob.geometry)
            if self.design.obstruction_shade_ratio > 0 and ob.height_m > 0:
                shift = poleward * self.design.obstruction_shade_ratio * ob.height_m
                g = unary_union([g, affinity.translate(g, 0, shift)]).convex_hull
            zones.append(g.buffer(self.rules.clearance_m(ob.kind), join_style="mitre"))
        return unary_union(zones) if zones else Polygon()

    def _flat_specs(self, lat: float) -> list[tuple[str, RowSpec]]:
        d = self.design
        t = math.radians(d.flat_tilt_deg)
        specs = []
        for orient in d.flat_orientations:
            along, slant = _module_dims(d.module, orient)
            depth = slant * math.cos(t)
            # GCR = module slant length / row pitch
            if d.flat_racking == Racking.EAST_WEST:
                pitch = slant / d.east_west_gcr
            elif d.flat_racking == Racking.SOUTH:
                pitch = slant / d.south_gcr if d.south_gcr else south_row_pitch_m(slant, d.flat_tilt_deg, lat)
            else:
                depth, pitch = slant, slant + d.module_gap_m
            specs.append((orient, RowSpec(along, depth, max(pitch, depth), d.module_gap_m)))
        return specs

    def _pack_flat(self, usable: BaseGeometry, lat: float) -> tuple[list[Polygon], dict]:
        if usable.is_empty:
            return [], {"racking": self.design.flat_racking.value}
        _short, _long, axis = principal_axes(usable)
        angles = [axis, (axis + 90.0) % 180.0]
        if self.design.flat_racking == Racking.SOUTH:
            # rows must run roughly east-west: pick the building axis nearest to it
            angles = [min(angles, key=lambda a: min(a, 180.0 - a))]
        gap_w = self.rules.section_gap_ft * FT
        max_sec = self.rules.max_array_section_ft * FT
        best: list[Polygon] = []
        best_info: dict = {}
        for angle in angles:
            aligned = rotate(usable, -angle)
            aligned = aligned.difference(section_gaps(aligned, max_sec, gap_w))
            for orient, spec in self._flat_specs(lat):
                placed = pack_rows(aligned, spec, self.design.phase_steps)
                if len(placed) > len(best):
                    best = [rotate(p, angle) for p in placed]
                    best_info = {
                        "racking": self.design.flat_racking.value,
                        "orientation": orient,
                        "row_axis_deg": round(angle, 1),
                        "row_pitch_m": round(spec.row_pitch_m, 3),
                    }
        return best, best_info

    def _pack_pitched(self, usable: BaseGeometry, plane: RoofPlane) -> tuple[list[Polygon], dict]:
        if usable.is_empty:
            return [], {"racking": Racking.FLUSH.value}
        stretch = 1.0 / math.cos(math.radians(plane.pitch_deg))
        # rotate so downslope points to +y, then unfold the slope along y
        surface = affinity.scale(rotate(usable, plane.azimuth_deg), 1.0, stretch, origin=(0, 0))
        g = self.design.module_gap_m
        best: list[Polygon] = []
        best_orient = None
        for orient in self.design.pitched_orientations:
            along, slant = _module_dims(self.design.module, orient)
            placed = pack_rows(surface, RowSpec(along, slant, slant + g, g), self.design.phase_steps)
            if len(placed) > len(best):
                best, best_orient = placed, orient
        plan = [rotate(affinity.scale(p, 1.0, 1.0 / stretch, origin=(0, 0)), -plane.azimuth_deg) for p in best]
        return plan, {"racking": Racking.FLUSH.value, "orientation": best_orient}
