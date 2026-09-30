"""Module packing.

Rows run along +x of the packing frame. Within a row, the x-ranges where the whole
row depth lies inside the usable polygon are computed exactly, then filled with as
many modules as fit. Several row phase offsets are tried and the best is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import clip_by_rect
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from .geometry import polygons

_SLIVER_M2 = 1e-4  # ignore overlap slivers smaller than 1 cm²


@dataclass(frozen=True)
class RowSpec:
    along_m: float  # module dimension along the row
    depth_m: float  # plan-view module dimension across the row
    row_pitch_m: float  # distance between row starts (>= depth_m)
    gap_m: float = 0.02  # gap between modules within a row


def _free_intervals(strip: Polygon, usable: BaseGeometry, x0: float, x1: float) -> list[tuple[float, float]]:
    sx0, sy0, sx1, sy1 = strip.bounds
    outside = strip.difference(clip_by_rect(usable, sx0, sy0, sx1, sy1))
    blocked = sorted((p.bounds[0], p.bounds[2]) for p in polygons(outside) if p.area > _SLIVER_M2)
    free, cursor = [], x0
    for a, b in blocked:
        if a > cursor:
            free.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < x1:
        free.append((cursor, x1))
    return free


def pack_rows(usable: BaseGeometry, spec: RowSpec, phase_steps: int = 8) -> list[Polygon]:
    """Return module rectangles (in the packing frame) that fit fully inside `usable`."""
    if usable.is_empty:
        return []
    minx, miny, maxx, maxy = usable.bounds
    x0, x1 = minx - 1.0, maxx + 1.0
    unit = spec.along_m + spec.gap_m
    best: list[Polygon] = []
    for k in range(phase_steps):
        placed: list[Polygon] = []
        y = miny + spec.row_pitch_m * k / phase_steps
        while y + spec.depth_m <= maxy + 1e-9:
            strip = box(x0, y, x1, y + spec.depth_m)
            for a, b in _free_intervals(strip, usable, x0, x1):
                n = int(math.floor((b - a + spec.gap_m) / unit + 1e-9))
                if n <= 0:
                    continue
                run = n * spec.along_m + (n - 1) * spec.gap_m
                start = a + (b - a - run) / 2
                placed.extend(
                    box(start + i * unit, y, start + i * unit + spec.along_m, y + spec.depth_m) for i in range(n)
                )
            y += spec.row_pitch_m
        if len(placed) > len(best):
            best = placed
    return best


def winter_profile_angle_deg(lat_deg: float, hour_angle_deg: float = 30.0) -> float:
    """Solar profile angle (in the north-south plane) at the winter solstice.

    The default hour angle of 30 degrees is 10:00/14:00 solar time, the common
    design window for inter-row shading on south-tilted arrays.
    """
    lat = math.radians(abs(lat_deg))
    dec = math.radians(-23.44)
    h = math.radians(hour_angle_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    alt = math.asin(max(min(sin_alt, 1.0), -1.0))
    if alt <= 0:
        raise ValueError(f"sun below horizon at design time for latitude {lat_deg}")
    cos_az = (math.sin(alt) * math.sin(lat) - math.sin(dec)) / (math.cos(alt) * math.cos(lat))
    az = math.acos(max(min(cos_az, 1.0), -1.0))  # from due south
    return math.degrees(math.atan(math.tan(alt) / math.cos(az)))


def south_row_pitch_m(slant_m: float, tilt_deg: float, lat_deg: float) -> float:
    """Row pitch for south-tilted racking with no inter-row shading at the design time."""
    t = math.radians(tilt_deg)
    profile = math.radians(winter_profile_angle_deg(lat_deg))
    return slant_m * math.cos(t) + slant_m * math.sin(t) / math.tan(profile)
