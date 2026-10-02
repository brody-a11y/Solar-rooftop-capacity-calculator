"""Free US geocoding from Overture Maps address points.

Overture's addresses theme republishes county and state address-point datasets
(via OpenAddresses), so a match usually sits on the parcel or building, not in
the street. Addresses need a house number and a 5-digit ZIP. The first lookup in
a ZIP scans the row groups whose postcode range covers it (~10-30 s) and caches
that ZIP's points on disk; later lookups in the same ZIP are instant.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .overture import BUCKET, default_filesystem, latest_release

_SUFFIX = {
    "STREET": "ST", "AVENUE": "AVE", "AV": "AVE", "ROAD": "RD", "BOULEVARD": "BLVD", "DRIVE": "DR",
    "LANE": "LN", "COURT": "CT", "PLACE": "PL", "PARKWAY": "PKWY", "HIGHWAY": "HWY", "CIRCLE": "CIR",
    "TERRACE": "TER", "TRAIL": "TRL", "SQUARE": "SQ", "PLAZA": "PLZ", "EXPRESSWAY": "EXPY",
    "FREEWAY": "FWY", "CENTER": "CTR", "CANYON": "CYN", "MOUNT": "MT", "SAINT": "ST",
}
_DIRECTION = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "NORTHEAST": "NE", "NORTHWEST": "NW",
              "SOUTHEAST": "SE", "SOUTHWEST": "SW"}
# Words that don't identify a street on their own. Abbreviated name words such
# as CYN (Canyon) or MT (Mount) are deliberately not in this set.
_GENERIC = {"ST", "AVE", "RD", "BLVD", "DR", "LN", "CT", "PL", "PKWY", "HWY", "CIR", "TER", "TRL", "SQ",
            "PLZ", "EXPY", "FWY", "WAY"} | set(_DIRECTION.values())


def street_tokens(street: str) -> list[str]:
    toks = re.findall(r"[A-Z0-9]+", street.upper())
    return [_DIRECTION.get(t, _SUFFIX.get(t, t)) for t in toks]


def parse_address(address: str) -> tuple[str, str, str] | None:
    """(house number, street, zip) from a one-line US address, or None."""
    zm = re.search(r"\b(\d{5})(?:-\d{4})?\s*(?:,?\s*(?:USA|US|United States))?\s*$", address.strip())
    nm = re.match(r"\s*(\d+[A-Za-z]?)(?:-\d+)?\s+([^,#]+)", address)
    if not zm or not nm:
        return None
    street = re.sub(r"\b(APT|UNIT|STE|SUITE|BLDG|BUILDING)\b.*$", "", nm.group(2), flags=re.IGNORECASE).strip()
    return nm.group(1).upper(), street, zm.group(1)


def _score(query: list[str], candidate: list[str]) -> float:
    """Share of the query's street-name words found in the candidate, requiring
    every distinctive (non-suffix, non-direction) word to match."""
    core = [t for t in query if t not in _GENERIC]
    cand = set(candidate)
    if not core or any(t not in cand for t in core):
        return 0.0
    return sum(t in cand for t in query) / len(query)


class OvertureAddresses:
    def __init__(self, cache_dir: str | Path = Path.home() / ".rooftop-solar" / "overture",
                 release: str | None = None, filesystem: pafs.FileSystem | None = None,
                 base_path: str | None = None, workers: int = 24):
        self.cache_dir = Path(cache_dir)
        self.fs = filesystem or default_filesystem()
        if base_path is None:
            self.release = release or latest_release(self.fs)
            base_path = f"{BUCKET}/release/{self.release}/theme=addresses/type=address"
        else:
            self.release = release or "local"
        self.base_path = base_path.rstrip("/")
        self.workers = workers
        self._index = None
        self._zip_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def _index_rows(self):
        with self._lock:
            if self._index is not None:
                return self._index
            path = self.cache_dir / f"address-index-{self.release}.json"
            if path.exists():
                self._index = json.loads(path.read_text())
                return self._index
            files = [i.path for i in self.fs.get_file_info(pafs.FileSelector(self.base_path))
                     if i.type == pafs.FileType.File and i.path.endswith(".parquet")]

            def one(p):
                with self.fs.open_input_file(p) as fh:
                    md = pq.ParquetFile(fh).metadata
                cols = {md.schema.column(i).path: i for i in range(md.num_columns)}
                rows = []
                for r in range(md.num_row_groups):
                    rg = md.row_group(r)
                    c, z = rg.column(cols["country"]).statistics, rg.column(cols["postcode"]).statistics
                    if c is None or not c.has_min_max or not (c.min <= "US" <= c.max):
                        continue
                    if z is None or not z.has_min_max:
                        continue
                    rows.append((p, r, z.min, z.max))
                return rows

            with ThreadPoolExecutor(max_workers=32) as pool:
                self._index = [row for part in pool.map(one, files) for row in part]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._index))
            return self._index

    def _zip_table(self, zipcode: str) -> pa.Table:
        with self._lock:
            lock = self._zip_locks.setdefault(zipcode, threading.Lock())
        with lock:
            cache = self.cache_dir / "addresses" / self.release / f"{zipcode}.parquet"
            if cache.exists():
                return pq.read_table(cache)
            groups = [(p, r) for p, r, lo, hi in self._index_rows() if lo <= zipcode <= hi]

            def scan(pr):
                with self.fs.open_input_file(pr[0]) as fh:
                    t = pq.ParquetFile(fh).read_row_group(pr[1], columns=["number", "street", "unit", "postcode", "country", "bbox"])
                t = t.filter(pc.and_(pc.equal(t["postcode"], zipcode), pc.equal(t["country"], "US")))
                return pa.table({
                    "number": t["number"], "street": t["street"], "unit": t["unit"],
                    "lat": pc.struct_field(t["bbox"], "ymin"), "lon": pc.struct_field(t["bbox"], "xmin"),
                })

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                parts = [t for t in pool.map(scan, groups) if t.num_rows]
            schema = pa.schema([("number", pa.string()), ("street", pa.string()), ("unit", pa.string()),
                                ("lat", pa.float64()), ("lon", pa.float64())])
            table = pa.concat_tables([p.cast(schema) for p in parts]) if parts else schema.empty_table()
            cache.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, cache)
            return table

    def lookup(self, address: str) -> tuple[float, float, str, str] | None:
        """(lat, lon, matched address, precision) or None.

        precision is "address_point" for an exact house-number match, or
        "interpolated" when the number is estimated between the nearest listed
        numbers on the same side of the same street (large complexes are often
        listed under other numbers)."""
        parsed = parse_address(address)
        if not parsed:
            return None
        number, street, zipcode = parsed
        table = self._zip_table(zipcode)
        if table.num_rows == 0:
            return None
        query = street_tokens(street)
        same_street: dict[str, list[dict]] = {}
        for r in table.to_pylist():
            name = r["street"] or ""
            if name not in same_street:
                same_street[name] = []
            same_street[name].append(r)
        scored = sorted(
            ((_score(query, street_tokens(name)), name) for name in same_street), reverse=True
        )
        if not scored or scored[0][0] == 0:
            return None
        name = scored[0][1]
        rows = same_street[name]
        exact = [r for r in rows if (r["number"] or "").upper() == number]
        if exact:
            best = min(exact, key=lambda r: r["unit"] is not None)
            return best["lat"], best["lon"], f"{best['number']} {name} {zipcode}", "address_point"
        return self._interpolate(number, rows, name, zipcode)

    @staticmethod
    def _interpolate(number: str, rows: list[dict], name: str, zipcode: str, max_gap: int = 200):
        digits = re.match(r"\d+", number)
        if not digits:
            return None
        n = int(digits.group())
        pts: dict[int, tuple[float, float]] = {}
        for r in rows:
            m = re.fullmatch(r"\d+", r["number"] or "")
            if m and int(m.group()) % 2 == n % 2:
                pts.setdefault(int(m.group()), (r["lat"], r["lon"]))
        lower = max((k for k in pts if k < n), default=None)
        upper = min((k for k in pts if k > n), default=None)
        if lower is None or upper is None or upper - lower > max_gap:
            return None
        f = (n - lower) / (upper - lower)
        (la0, lo0), (la1, lo1) = pts[lower], pts[upper]
        return la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0), f"~{n} {name} {zipcode} (between {lower} and {upper})", "interpolated"

    def unit_points(self, address: str) -> list[tuple[float, float]]:
        """(lat, lon) of every address point sharing this house number and street
        (apartment units, building letters), or every number of a range such as
        "521-537 Edgewood Ave". Several distinct points usually mean a
        multi-building property."""
        return sorted({(round(r["lat"], 6), round(r["lon"], 6)) for r in self.address_records(address)})

    def address_records(self, address: str) -> list[dict]:
        """Every address record behind unit_points: one per unit where the
        county lists units, so len() is a unit count when units are listed."""
        parsed = parse_address(address)
        if not parsed:
            return []
        number, street, zipcode = parsed
        table = self._zip_table(zipcode)
        if table.num_rows == 0:
            return []
        rng = re.match(r"\s*(\d+)\s*-\s*(\d+)\s", address)
        if rng and int(rng.group(1)) < int(rng.group(2)) <= int(rng.group(1)) + 400:
            # "521-537 Edgewood Ave": every house number in the range
            lo, hi = int(rng.group(1)), int(rng.group(2))
            wanted = {str(n) for n in range(lo, hi + 1)}
            rows = [r for r in table.to_pylist() if str(r["number"] or "").upper() in wanted]
        else:
            rows = table.filter(pc.equal(pc.utf8_upper(table["number"]), number)).to_pylist()
        query = street_tokens(street)
        scored = [(_score(query, street_tokens(r["street"] or "")), r) for r in rows]
        top = max((sc for sc, _ in scored), default=0.0)
        if top == 0:
            return []
        return [r for sc, r in scored if sc == top]

    def homes_in(self, zipcode: str, footprints: list, margin_m: float = 3.0) -> list[int]:
        """Distinct addresses (number, street, unit) on or within `margin_m` of each
        lon/lat footprint: the homes in a townhome row, 1 for a house. 0 where
        the address data has none."""
        import shapely
        from shapely.geometry import Polygon

        from ..geometry import LocalFrame

        table = self._zip_table(zipcode) if zipcode else None
        if table is None or table.num_rows == 0 or not footprints:
            return [0] * len(footprints)
        lon, lat = table["lon"].to_numpy(zero_copy_only=False), table["lat"].to_numpy(zero_copy_only=False)
        keys = list(zip(table["number"].to_pylist(), table["street"].to_pylist(), table["unit"].to_pylist()))
        out = []
        for fp in footprints:
            frame = LocalFrame.for_geometry(fp)
            area = frame.to_lonlat(frame.to_local(fp).buffer(margin_m))
            x0, y0, x1, y1 = area.bounds
            near = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
            idx = [i for i in near.nonzero()[0] if shapely.contains_xy(area, lon[i], lat[i])]
            out.append(len({keys[i] for i in idx}))
        return out
