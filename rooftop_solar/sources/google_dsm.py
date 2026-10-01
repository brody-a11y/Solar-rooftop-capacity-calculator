"""Rooftop equipment (condensers, RTUs, vents) from Google's surface model.

Google's buildingInsights panel layout often runs straight through fields of
small condensing units. The Solar API dataLayers endpoint returns a digital
surface model (DSM, heights in metres) for the area around a point; objects that
stand above the surrounding roof are found with a morphological top-hat (roof
minus its grey opening) and returned as plan-view polygons. The sizing code
keeps panels clear of them (DEFAULT_CLEARANCES_FT["equipment"]).

Billed per dataLayers request (a separate Solar API SKU from buildingInsights);
results are cached on disk as small GeoJSON files, not the raster.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union

from ..geometry import LocalFrame, polygons

DATA_LAYERS_URL = "https://solar.googleapis.com/v1/dataLayers:get"


def detect_equipment(dsm: np.ndarray, roof_mask: np.ndarray, pixel_m: float, min_height_m: float = 0.3,
                     max_size_m: float = 6.0, min_area_m2: float = 0.4) -> list[np.ndarray]:
    """Boolean masks of objects standing at least `min_height_m` above the roof
    and no wider than about `max_size_m` (larger raised areas are roof levels or
    penthouses, which Google's own layout already avoids)."""
    from scipy import ndimage

    roof = roof_mask & np.isfinite(dsm)
    if not roof.any():
        return []
    k = max(3, int(round(max_size_m / pixel_m)) | 1)
    # Opening restricted to the roof: erode with off-roof pixels ignored, then
    # dilate without letting off-roof pixels win.
    eroded = ndimage.grey_erosion(np.where(roof, dsm, np.inf), size=(k, k))
    base = ndimage.grey_dilation(np.where(roof, eroded, -np.inf), size=(k, k))
    tall = roof & (dsm - base >= min_height_m)
    labels, n = ndimage.label(tall)
    if n == 0:
        return []
    min_px = max(1, int(math.ceil(min_area_m2 / (pixel_m * pixel_m))))
    sizes = ndimage.sum(tall, labels, index=np.arange(1, n + 1))
    return [labels == i + 1 for i, s in enumerate(sizes) if s >= min_px]


def mask_to_polygon(mask: np.ndarray, pixel_to_xy) -> Polygon:
    """Union of the pixel squares in `mask`; pixel_to_xy(col, row) gives a pixel corner."""
    boxes = []
    for r in np.unique(np.nonzero(mask)[0]):
        cols = np.nonzero(mask[r])[0]
        starts = cols[np.r_[True, np.diff(cols) > 1]]
        ends = cols[np.r_[np.diff(cols) > 1, True]] + 1
        for c0, c1 in zip(starts, ends):
            (x0, y0), (x1, y1) = pixel_to_xy(c0, r), pixel_to_xy(c1, r + 1)
            boxes.append(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return unary_union(boxes)


def read_geotiff(data: bytes):
    """(array, pixel_to_crs(col, row), epsg) from a north-up GeoTIFF."""
    import tifffile

    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        page = tif.pages[0]
        arr = page.asarray().astype("float64")
        if arr.ndim == 3:
            arr = arr[..., 0]
        tags = page.tags
        scale = tags["ModelPixelScaleTag"].value
        tie = tags["ModelTiepointTag"].value
        keys = tags["GeoKeyDirectoryTag"].value if "GeoKeyDirectoryTag" in tags else ()
        nodata = tags["GDAL_NODATA"].value if "GDAL_NODATA" in tags else None
    if nodata not in (None, ""):
        arr[arr == float(str(nodata).strip("\x00"))] = np.nan
    i0, j0, x0, y0 = tie[0], tie[1], tie[3], tie[4]
    sx, sy = scale[0], scale[1]
    epsg = None
    for n in range(4, len(keys), 4):  # GeoKey entries: id, location, count, value
        if keys[n] in (3072, 2048) and keys[n + 1] == 0:  # projected / geographic CRS code
            epsg = int(keys[n + 3])
            if keys[n] == 3072:
                break

    def pixel_to_crs(col, row):
        return x0 + (col - i0) * sx, y0 - (row - j0) * sy

    return arr, pixel_to_crs, epsg or 4326


class GoogleDSMClient:
    def __init__(self, api_key: str, cache_dir: str | Path | None = Path.home() / ".rooftop-solar" / "equipment_cache",
                 pixel_size_m: float = 0.25, timeout: float = 60.0, session: requests.Session | None = None):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pixel_size_m = pixel_size_m
        self.timeout = timeout
        self.session = session or requests.Session()

    def _download(self, footprint: Polygon) -> bytes:
        c = footprint.centroid
        frame = LocalFrame.for_geometry(footprint)
        local = frame.to_local(footprint)
        cx, cy = frame.point_to_local(c.x, c.y)
        radius = max(math.hypot(x - cx, y - cy) for x, y in local.exterior.coords) + 3.0
        resp = self.session.get(DATA_LAYERS_URL, params={
            "location.latitude": f"{c.y:.7f}", "location.longitude": f"{c.x:.7f}",
            "radiusMeters": f"{min(radius, 100.0):.0f}", "view": "DSM_LAYER",
            "requiredQuality": "MEDIUM", "pixelSizeMeters": f"{self.pixel_size_m:g}", "key": self.api_key,
        }, timeout=self.timeout)
        resp.raise_for_status()
        url = resp.json().get("dsmUrl")
        if not url:
            raise ValueError("no_dsm_in_response")
        tif = self.session.get(url, params={"key": self.api_key}, timeout=self.timeout)
        tif.raise_for_status()
        return tif.content

    def equipment(self, footprint: Polygon, min_height_m: float = 0.3, max_size_m: float = 6.0) -> list[Polygon]:
        """Plan-view (lon/lat) outlines of rooftop equipment on this footprint."""
        key = hashlib.sha1(f"{footprint.wkb_hex}|{self.pixel_size_m}|{min_height_m}|{max_size_m}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            return [shape(g) for g in json.loads(path.read_text())]
        arr, pixel_to_crs, epsg = read_geotiff(self._download(footprint))
        to_crs = Transformer.from_crs(4326, epsg, always_xy=True)
        to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
        fp_crs = Polygon([to_crs.transform(x, y) for x, y in footprint.exterior.coords])
        rows, cols = np.indices(arr.shape)
        cx, cy = pixel_to_crs(cols + 0.5, rows + 0.5)
        import shapely

        roof = shapely.contains_xy(fp_crs, cx, cy)
        pixel_m = abs(pixel_to_crs(1, 0)[0] - pixel_to_crs(0, 0)[0])
        if epsg == 4326:  # degrees: convert to metres for the size thresholds
            pixel_m *= 111_320 * math.cos(math.radians(footprint.centroid.y))
        out = []
        for m in detect_equipment(arr, roof, pixel_m, min_height_m, max_size_m):
            g = mask_to_polygon(m, pixel_to_crs)
            for part in polygons(g):
                out.append(Polygon([to_ll.transform(x, y) for x, y in part.exterior.coords]))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([mapping(g) for g in out]))
        return out
