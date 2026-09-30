import csv

import pytest

from rooftop_solar import cli
from rooftop_solar.accuracy import _best_error, load_truth
from rooftop_solar.sources.google_solar import GoogleLookupError, GoogleSolarClient

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
    assert float(row["google_unclipped_kw"]) >= float(row["google_kw"])


def test_sample_points_spread_over_large_footprint():
    from rooftop_solar.sources.google_solar import sample_points
    from .helpers import lonlat_box

    fp = lonlat_box(0, 0, 200, 100)
    pts = sample_points(fp, spacing_m=35, max_points=9)
    assert len(pts) == 9
    local = [FRAME.point_to_local(*p) for p in pts]
    assert all(0 < x < 200 and 0 < y < 100 for x, y in local)
    assert len(sample_points(lonlat_box(0, 0, 20, 20), max_points=9)) == 1


def test_fetch_building_merges_split_buildings_and_skips_neighbours():
    from rooftop_solar.sources.google_solar import fetch_building
    from .helpers import lonlat_box

    fp = lonlat_box(0, 0, 120, 40)

    class FakeClient:
        calls = 0

        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            self.calls += 1
            x, _ = FRAME.point_to_local(lon, lat)
            if x < 60:
                name, cx = "buildings/west", 30
            else:
                name, cx = "buildings/east", 90
            clon, clat = FRAME.point_to_lonlat(cx, 20)
            r = _fake_insights(clat, clon)
            r["name"], r["center"] = name, {"latitude": clat, "longitude": clon}
            return r

    client = FakeClient()
    merged = fetch_building(client, fp, max_points=9)
    assert merged.buildings_merged == 2
    assert len(merged.panels) == 120 and len(merged.segments) == 2
    assert {p.segment_index for p in merged.panels} == {0, 1}

    class NeighbourClient(FakeClient):
        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            clon, clat = FRAME.point_to_lonlat(500, 500)  # a building far away
            r = _fake_insights(clat, clon)
            r["name"] = "buildings/neighbour"
            return r

    with pytest.raises(GoogleLookupError, match="no_google_building"):
        fetch_building(NeighbourClient(), fp)


def test_fetch_building_retries_low_quality_after_404s():
    import requests
    from rooftop_solar.sources.google_solar import fetch_building
    from .helpers import lonlat_box

    fp = lonlat_box(0, 0, 40, 40)
    seen = []

    class Client:
        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            seen.append(required_quality)
            if required_quality == "MEDIUM":
                resp = requests.Response()
                resp.status_code = 404
                raise requests.HTTPError(response=resp)
            r = _fake_insights(lat, lon)
            r["imageryQuality"] = "LOW"
            return r

    ins = fetch_building(Client(), fp)
    assert seen[-1] == "LOW" and ins.imagery_quality == "LOW"

    class Always403:
        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            resp = requests.Response()
            resp.status_code = 403
            raise requests.HTTPError(response=resp)

    with pytest.raises(GoogleLookupError, match="http_403"):
        fetch_building(Always403(), fp)


def test_footprint_from_insights_wraps_panel_grid():
    from rooftop_solar.sources.google_solar import GoogleInsights, footprint_from_insights

    lon, lat = ll(0, 0)
    ins = GoogleInsights.from_response(_fake_insights(lat, lon))
    fp = FRAME.to_local(footprint_from_insights(ins, pad_m=1.6))
    # 10 x 6 grid of 1.879 x 1.045 m panels on a 2.0 x 1.1 m pitch -> ~19.9 x 6.5 m, padded 1.6 m each side
    minx, miny, maxx, maxy = fp.bounds
    assert maxx - minx == pytest.approx(19.879 + 3.2, abs=0.2)
    assert maxy - miny == pytest.approx(6.545 + 3.2, abs=0.2)


def test_site_without_mapped_building_falls_back_to_google(tmp_path, footprints, monkeypatch):
    monkeypatch.setattr(cli, "OvertureFootprints", lambda **kw: footprints)
    monkeypatch.setattr(GoogleSolarClient, "building_insights", lambda self, lat, lon, required_quality="MEDIUM": _fake_insights(lat, lon))
    monkeypatch.setenv("GOOGLE_SOLAR_API_KEY", "test")
    lon, lat = ll(900, 900)  # nothing in the footprint data here
    truth = tmp_path / "truth.csv"
    truth.write_text(f"name,address,latitude,longitude,true_kw,module_w\nNew build,,{lat},{lon},20,550\n")
    out = tmp_path / "acc.csv"
    assert cli.main(["accuracy", "--truth", str(truth), "--out", str(out), "--google", "--google-cache", str(tmp_path / "gc"), "--workers", "1"]) == 0
    with out.open() as f:
        row = list(csv.DictReader(f))[0]
    assert "footprint_from_google_imagery_check_date" in row["reasons"]
    assert float(row["google_kw"]) > 0
    assert row["google_imagery_date"] == "2025-01-01"
