import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from rooftop_solar.sources.overture import OvertureFootprints

from .helpers import FRAME

# Local stand-in for the Overture buildings dataset: (id, local-metre box, class)
BUILDINGS = [
    ("warehouse", (0, 0, 60, 40), "warehouse"),
    ("apt-a", (200, 0, 240, 15), "apartments"),
    ("apt-b", (200, 30, 240, 45), "apartments"),  # 15 m north of apt-a, same complex
    ("unknown", (400, 0, 430, 20), None),
]


def _write_dataset(root):
    rows = {"id": [], "geometry": [], "bbox": [], "height": [], "num_floors": [], "class": [], "subtype": [], "roof_shape": []}
    for bid, b, cls in BUILDINGS:
        g = FRAME.to_lonlat(box(*b))
        x0, y0, x1, y1 = g.bounds
        rows["id"].append(bid)
        rows["geometry"].append(g.wkb)
        rows["bbox"].append({"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1})
        rows["height"].append(None)
        rows["num_floors"].append(None)
        rows["class"].append(cls)
        rows["subtype"].append(None)
        rows["roof_shape"].append(None)
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), root / "part-0.parquet", row_group_size=2)


@pytest.fixture
def footprints(tmp_path):
    data = tmp_path / "buildings"
    _write_dataset(data)
    return OvertureFootprints(cache_dir=tmp_path / "cache", release="test", filesystem=pafs.LocalFileSystem(), base_path=str(data))



@pytest.fixture(autouse=True)
def _isolate_from_real_credentials(tmp_path, monkeypatch):
    """Self-tests run on users' machines (Install.command): never read their saved
    Regrid token or Google key, so tests can't make billed calls."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("REGRID_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_SOLAR_API_KEY", raising=False)
    # Equipment detection is on by default with --google: tests that fake Google
    # must not reach the real dataLayers endpoint either.
    from rooftop_solar.sources import google_dsm

    def _no_network(self, footprint):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(google_dsm.GoogleDSMClient, "_download", _no_network)
