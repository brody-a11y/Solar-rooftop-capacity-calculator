import csv

import pytest

from rooftop_solar import cli
from rooftop_solar.accuracy import _best_error, load_truth
from rooftop_solar.sources.google_solar import GoogleSolarClient

from .helpers import FRAME, ll


def test_load_truth_csv_groups_designs(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "name,address,latitude,longitude,true_kw,module_w\n"
        "A,1 Main St,34.0,-118.0,100,550\n"
        "A,1 Main St,34.0,-118.0,120,\n"
        "B,2 Main St,,,50,600\n"
    )
    t = load_truth(p)
    assert [d["kw"] for d in t["A"]["truths"]] == [100.0, 120.0]
    assert t["A"]["lat"] == 34.0 and t["B"]["lat"] is None


def test_best_error_picks_closest_design_after_module_scaling():
    err, design = _best_error(100.0, [{"kw": 150, "module_w": 600}, {"kw": 112, "module_w": 600}], module_w=500)
    assert design["kw"] == 112  # 100 kW at 500 W -> 120 kW at 600 W
    assert err == pytest.approx(120 / 112 - 1)


def _fake_insights(lat, lon):
    """Google-like response: a 6 x 10 grid of flat panels centred on the point."""
    x0, y0 = FRAME.point_to_local(lon, lat)
    panels = []
    for i in range(10):
        for j in range(6):
            plon, plat = FRAME.point_to_lonlat(x0 - 10 + i * 2.0, y0 - 6 + j * 1.1)
            panels.append({"center": {"latitude": plat, "longitude": plon}, "orientation": "LANDSCAPE", "segmentIndex": 0})
    return {
        "imageryQuality": "HIGH",
        "imageryDate": {"year": 2025, "month": 1, "day": 1},
        "solarPotential": {
            "maxArrayPanelsCount": len(panels), "panelCapacityWatts": 400, "panelHeightMeters": 1.879,
            "panelWidthMeters": 1.045, "roofSegmentStats": [{"pitchDegrees": 1.0, "azimuthDegrees": 180.0}],
            "solarPanels": panels,
        },
    }


def test_accuracy_command_with_google(tmp_path, footprints, monkeypatch):
    monkeypatch.setattr(cli, "OvertureFootprints", lambda **kw: footprints)
    monkeypatch.setattr(GoogleSolarClient, "building_insights", lambda self, lat, lon, required_quality="MEDIUM": _fake_insights(lat, lon))
    monkeypatch.setenv("GOOGLE_SOLAR_API_KEY", "test")
    lon, lat = ll(30, 20)
    truth = tmp_path / "truth.csv"
    truth.write_text(f"name,address,latitude,longitude,true_kw,module_w\nWarehouse,,{lat},{lon},30,550\n")
    out = tmp_path / "acc.csv"
    assert cli.main(["accuracy", "--truth", str(truth), "--out", str(out), "--google", "--google-cache", str(tmp_path / "gc"),
                     "--layouts", str(tmp_path / "a.kml"), "--workers", "1"]) == 0
    with out.open() as f:
        row = list(csv.DictReader(f))[0]
    assert row["google_kw"] and float(row["google_kw"]) > 0
    assert row["footprint_only_kw"] and float(row["footprint_only_kw"]) > float(row["google_kw"])
    assert row["tool_kw"] == row["google_kw"]  # Google result is primary when usable
