"""Building footprints from Overture Maps (free, open data on AWS S3).

Overture's buildings theme merges OpenStreetMap, Microsoft and Google Open
Buildings footprints. It is ~500 GeoParquet files, so a direct spatial query
takes most of a minute. Instead, a one-time index of every row group's bounding
box is built from the file footers (~10 s, cached per release), and each lookup
reads only the row groups that overlap the point. Row groups read are cached on
disk, so re-runs over the same area are fast.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon

from ..geometry import LocalFrame
from ..models import Occupancy

BUCKET = "overturemaps-us-west-2"
_COLUMNS = ["id", "geometry", "bbox", "height", "num_floors", "class", "subtype", "roof_shape"]

# Overture building `class` -> occupancy. Classes not listed give None (unknown).
_R2 = {"apartments", "dormitory"}
_R3 = {"house", "detached", "semidetached_house", "terrace", "bungalow", "cabin", "farm", "static_caravan"}
_COMMERCIAL = {
    "commercial", "retail", "industrial", "warehouse", "office", "supermarket", "hotel", "school",
    "university", "college", "hospital", "public", "civic", "government", "church", "manufacture",
    "kindergarten", "parking", "service", "sports_hall", "storage_tank", "transportation", "garage",
}


def occupancy_from_class(cls: str | None) -> Occupancy | None:
    if cls in _R2:
        return Occupancy.R2
    if cls in _R3:
        return Occupancy.R3
    if cls in _COMMERCIAL:
        return Occupancy.COMMERCIAL
    return None


@dataclass
class FootprintMatch:
    overture_id: str
    footprint: Polygon
    distance_m: float  # 0 when the point is inside the footprint
    building_class: str | None
    subtype: str | None
    height_m: float | None
    num_floors: int | None
    roof_shape: str | None
    runner_up_m: float | None = None  # distance to the next-closest building


def default_filesystem() -> pafs.FileSystem:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return pafs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=proxy or None)


def latest_release(fs: pafs.FileSystem) -> str:
    infos = fs.get_file_info(pafs.FileSelector(f"{BUCKET}/release"))
    releases = sorted(Path(i.path).name for i in infos if i.type == pafs.FileType.Directory)
    if not releases:
        raise RuntimeError("no Overture releases found")
    return releases[-1]


def _largest(geom) -> Polygon | None:
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom if isinstance(geom, Polygon) else None


class OvertureFootprints:
    def __init__(
        self,
        cache_dir: str | Path = Path.home() / ".rooftop-solar" / "overture",
        release: str | None = None,
        filesystem: pafs.FileSystem | None = None,
        base_path: str | None = None,
        workers: int = 8,
    ):
        self.cache_dir = Path(cache_dir)
        self.fs = filesystem or default_filesystem()
        if base_path is None:
            self.release = release or latest_release(self.fs)
            base_path = f"{BUCKET}/release/{self.release}/theme=buildings/type=building"
        else:
            self.release = release or "local"
        self.base_path = base_path.rstrip("/")
        self.workers = workers
        self._index: list[tuple[str, int, float, float, float, float]] | None = None
        self._lock = threading.Lock()

    # -- index ----------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.cache_dir / f"index-{self.release}.json"

    def index(self) -> list[tuple[str, int, float, float, float, float]]:
        with self._lock:
            if self._index is not None:
                return self._index
            path = self._index_path()
            if path.exists():
                self._index = [tuple(r) for r in json.loads(path.read_text())]
                return self._index
            files = [
                i.path
                for i in self.fs.get_file_info(pafs.FileSelector(self.base_path))
                if i.type == pafs.FileType.File and i.path.endswith(".parquet")
            ]
            with ThreadPoolExecutor(max_workers=32) as pool:
                parts = list(pool.map(self._file_index, files))
            self._index = [row for part in parts for row in part]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._index))
            return self._index

    def _file_index(self, path: str) -> list[tuple[str, int, float, float, float, float]]:
        with self.fs.open_input_file(path) as fh:
            md = pq.ParquetFile(fh).metadata
        cols = {md.schema.column(i).path: i for i in range(md.num_columns)}
        rows = []
        for r in range(md.num_row_groups):
            rg = md.row_group(r)
            s = {k: rg.column(cols[f"bbox.{k}"]).statistics for k in ("xmin", "xmax", "ymin", "ymax")}
            rows.append((path, r, s["xmin"].min, s["xmax"].max, s["ymin"].min, s["ymax"].max))
        return rows

    # -- lookup ---------------------------------------------------------------

    def _row_group(self, path: str, rg: int):
        cache = self.cache_dir / "rowgroups" / self.release / f"{Path(path).stem}-{rg}.parquet"
        if cache.exists():
            return pq.read_table(cache)
        with self.fs.open_input_file(path) as fh:
            tbl = pq.ParquetFile(fh).read_row_group(rg, columns=_COLUMNS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, cache)
        return tbl

    def find(
        self, points: dict[str, tuple[float, float]], search_m: float = 40.0, campus_m: float = 0.0
    ) -> dict[str, list[FootprintMatch]]:
        """For each key -> (lon, lat), return the building at the point, or the
        nearest within `search_m`. With `campus_m` > 0, also every building whose
        centre is within `campus_m` of the point (multi-building properties).
        The first match in each list is the primary building."""
        index = self.index()
        ext = np.array([r[2:] for r in index], dtype=float).reshape(-1, 4)  # xmin, xmax, ymin, ymax
        reach = max(search_m, campus_m)
        boxes = {}
        needed: dict[tuple[str, int], list[str]] = {}
        for key, (lon, lat) in points.items():
            dlat = reach / 111_320.0
            dlon = reach / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
            b = (lon - dlon, lon + dlon, lat - dlat, lat + dlat)
            boxes[key] = b
            hit = (ext[:, 0] <= b[1]) & (ext[:, 1] >= b[0]) & (ext[:, 2] <= b[3]) & (ext[:, 3] >= b[2])
            for i in np.flatnonzero(hit):
                needed.setdefault((index[i][0], index[i][1]), []).append(key)

        candidates: dict[str, list[dict]] = {k: [] for k in points}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            tables = dict(zip(needed, pool.map(lambda pr: self._row_group(*pr), needed)))
        for (path, rg), keys in needed.items():
            tbl = tables[(path, rg)]
            bb = tbl.column("bbox").to_pylist()
            for key in keys:
                x0, x1, y0, y1 = boxes[key]
                hits = [i for i, b in enumerate(bb) if b["xmin"] <= x1 and b["xmax"] >= x0 and b["ymin"] <= y1 and b["ymax"] >= y0]
                if hits:
                    candidates[key].extend(tbl.take(hits).to_pylist())

        return {key: self._match(points[key], candidates[key], search_m, campus_m) for key in points}

    def _match(self, lonlat, rows, search_m, campus_m) -> list[FootprintMatch]:
        frame = LocalFrame(*lonlat)
        pt = Point(0, 0)
        seen, matches = set(), []
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            poly = _largest(shapely.from_wkb(row["geometry"]))
            if poly is None:
                continue
            local = frame.to_local(poly)
            matches.append(
                (
                    local.distance(pt),
                    local.centroid.distance(pt),
                    FootprintMatch(
                        row["id"], poly, 0.0, row.get("class"), row.get("subtype"),
                        row.get("height"), row.get("num_floors"), row.get("roof_shape"),
                    ),
                )
            )
        if not matches:
            return []
        matches.sort(key=lambda m: (m[0], m[1]))
        dist, _c, primary = matches[0]
        if dist > search_m:
            return []
        primary.distance_m = dist
        if len(matches) > 1:
            primary.runner_up_m = matches[1][0]
        out = [primary]
        if campus_m > 0:
            for d, c, m in matches[1:]:
                if c <= campus_m and frame.to_local(m.footprint).area >= 50.0:
                    m.distance_m = d
                    out.append(m)
        return out


    def in_polygons(self, areas: dict, min_overlap: float = 0.5) -> dict[str, list[FootprintMatch]]:
        """Buildings lying mostly (>= `min_overlap` of their area) inside each
        key -> lon/lat polygon, e.g. a parcel. Largest building first."""
        points, radius = {}, {}
        for key, poly in areas.items():
            c = poly.representative_point()
            frame = LocalFrame(c.x, c.y)
            local = frame.to_local(poly)
            points[key] = (c.x, c.y)
            radius[key] = max(Point(0, 0).distance(Point(xy)) for g in getattr(local, "geoms", [local]) for xy in g.exterior.coords)
        out: dict[str, list[FootprintMatch]] = {}
        for key, poly in areas.items():
            r = radius[key] + 5.0
            found = self.find({key: points[key]}, search_m=r, campus_m=r)[key]
            frame = LocalFrame(*points[key])
            local_poly = frame.to_local(poly)
            keep = []
            for m in found:
                fp = frame.to_local(m.footprint)
                if fp.area > 0 and fp.intersection(local_poly).area >= min_overlap * fp.area:
                    keep.append((fp.area, m))
            out[key] = [m for _a, m in sorted(keep, key=lambda t: -t[0])]
        return out
