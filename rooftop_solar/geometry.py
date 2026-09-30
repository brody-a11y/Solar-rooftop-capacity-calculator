"""Projection and polygon helpers. All sizing math runs in a local metric frame."""

from __future__ import annotations

import math

from pyproj import Transformer
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


class LocalFrame:
    """Azimuthal equidistant projection centred on a point: x = east (m), y = north (m)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        proj = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m"
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    @classmethod
    def for_geometry(cls, geom: BaseGeometry) -> "LocalFrame":
        c = geom.centroid
        return cls(c.x, c.y)

    def to_local(self, geom: BaseGeometry) -> BaseGeometry:
        return transform(self._fwd.transform, geom)

    def to_lonlat(self, geom: BaseGeometry) -> BaseGeometry:
        return transform(self._inv.transform, geom)

    def point_to_local(self, lon: float, lat: float) -> tuple[float, float]:
        return self._fwd.transform(lon, lat)

    def point_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return self._inv.transform(x, y)


def principal_axes(poly: BaseGeometry) -> tuple[float, float, float]:
    """(short side, long side, angle of long side in degrees CCW from +x) of the
    minimum rotated rectangle."""
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    edges = []
    for (x1, y1), (x2, y2) in zip(coords[:2], coords[1:3]):
        edges.append((math.hypot(x2 - x1, y2 - y1), math.degrees(math.atan2(y2 - y1, x2 - x1))))
    edges.sort()
    (short, _), (long_, angle) = edges
    return short, long_, angle % 180.0


def polygons(geom: BaseGeometry) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for sub in geom.geoms for g in polygons(sub)]
    return []


def inset(poly: BaseGeometry, distance_m: float) -> BaseGeometry:
    """Inward offset with mitred corners so rectangular roofs stay rectangular."""
    if distance_m <= 0:
        return poly
    return poly.buffer(-distance_m, join_style="mitre")


def section_gaps(poly: BaseGeometry, max_section_m: float, gap_m: float) -> BaseGeometry:
    """Pathway strips that split `poly` (already in its packing frame) into array
    sections no larger than `max_section_m` along x and y."""
    minx, miny, maxx, maxy = poly.bounds
    strips = []
    for lo, hi, vertical in ((minx, maxx, True), (miny, maxy, False)):
        extent = hi - lo
        if extent <= max_section_m:
            continue
        n_sections = math.ceil((extent + gap_m) / (max_section_m + gap_m))
        section = (extent - (n_sections - 1) * gap_m) / n_sections
        for i in range(1, n_sections):
            start = lo + i * section + (i - 1) * gap_m
            if vertical:
                strips.append(box(start, miny - 1, start + gap_m, maxy + 1))
            else:
                strips.append(box(minx - 1, start, maxx + 1, start + gap_m))
    if not strips:
        return Polygon()
    return MultiPolygon(strips).buffer(0)


def rotate(geom: BaseGeometry, angle_deg: float) -> BaseGeometry:
    return affinity.rotate(geom, angle_deg, origin=(0, 0))
