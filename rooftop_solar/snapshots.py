"""Satellite snapshots of sized sites, for checking layouts by eye.

One PNG per site: Google's aerial image (Solar API dataLayers, one billed call
per site, cached) with the roof outlines (yellow), counted panels (blue),
detected rooftop equipment (red) and raised-racking areas (orange) drawn on it.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform, unary_union

from .geometry import LocalFrame, polygons
from .sources.google_dsm import read_geotiff

YELLOW, BLUE, RED, ORANGE = (255, 221, 0), (30, 110, 255), (255, 30, 30), (255, 140, 0)


def _paint(img: np.ndarray, geom, cx: np.ndarray, cy: np.ndarray, color, alpha: float, outline_m: float = 0.0):
    """Blend `color` into img where (cx, cy) falls in geom (or its outline band)."""
    if geom is None or geom.is_empty:
        return
    if outline_m:
        geom = geom.boundary.buffer(outline_m)
    minx, miny, maxx, maxy = geom.bounds
    sel = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    if not sel.any():
        return
    rows, cols = np.nonzero(sel)
    inside = shapely.contains_xy(geom, cx[rows, cols], cy[rows, cols])
    r, c = rows[inside], cols[inside]
    img[r, c] = (1 - alpha) * img[r, c] + alpha * np.array(color, dtype=float)


def render(tif: bytes, layers: list[tuple[list[Polygon], tuple, float, float]]) -> np.ndarray:
    """RGB uint8 array: the GeoTIFF imagery with lon/lat layers (polys, color, alpha, outline_m) drawn on."""
    arr, pixel_to_crs, epsg = read_geotiff(tif, raw=True)
    img = np.asarray(arr)[..., :3].astype(float)
    if img.max() <= 1.0:
        img *= 255.0
    rows, cols = np.indices(img.shape[:2])
    cx, cy = pixel_to_crs(cols + 0.5, rows + 0.5)
    to_crs = Transformer.from_crs(4326, epsg, always_xy=True).transform
    for polys, color, alpha, outline in layers:
        geoms = [transform(to_crs, p) for p in polys if p is not None and not p.is_empty]
        if geoms:
            _paint(img, unary_union(geoms), cx, cy, color, alpha, outline)
    return img.clip(0, 255).astype(np.uint8)


def site_layers(outcome) -> tuple[list[Polygon], list]:
    """(footprints, layers) for one SiteOutcome's counted buildings."""
    est = outcome._counted()
    footprints = [b.footprint for b, e, c in outcome._triples() if c]
    prim = [e.primary for e in est if e.primary]
    layers = [
        ([p for r in prim for p in r.layout], BLUE, 0.55, 0.0),
        ([p for r in prim for p in r.raised_areas], ORANGE, 0.35, 0.0),
        ([p for r in prim for p in r.equipment], RED, 1.0, 0.15),
        (footprints, YELLOW, 1.0, 0.25),
    ]
    return footprints, layers


def snapshot(outcome, client, out_dir: str | Path, cache_dir: str | Path | None = None) -> Path | None:
    """Write <out_dir>/<site>.png; returns the path, or None if nothing to draw."""
    footprints, layers = site_layers(outcome)
    if not footprints:
        return None
    allfp = unary_union(footprints)
    frame = LocalFrame.for_geometry(allfp)
    local = frame.to_local(allfp)
    c = local.centroid
    radius = max(math.hypot(x - c.x, y - c.y) for g in polygons(local) for x, y in g.exterior.coords) + 10.0
    lon, lat = frame.point_to_lonlat(c.x, c.y)
    key = hashlib.sha1(f"{lat:.6f},{lon:.6f},{min(radius, 100.0):.0f}".encode()).hexdigest()
    cache = Path(cache_dir) / f"{key}.tif" if cache_dir else None
    if cache and cache.exists():
        tif = cache.read_bytes()
    else:
        tif = client.imagery(lat, lon, radius)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(tif)
    import imagecodecs

    out = Path(out_dir) / (re.sub(r"[^\w.-]+", "_", outcome.site.id).strip("_") + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(imagecodecs.png_encode(render(tif, layers)))
    return out
