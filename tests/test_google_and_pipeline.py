import csv
import json
import random

import pytest

from rooftop_solar import Building, DesignConfig, GeometricEstimator, Module, Occupancy, Racking
from rooftop_solar.calibration import Calibrator, Sample, accuracy, cross_validate
from rooftop_solar.cli import main
from rooftop_solar.models import FT
from rooftop_solar.pipeline import estimate_site
from rooftop_solar.sources.google_solar import GoogleFilteredEstimator, GoogleInsights

from .helpers import FRAME, centered_box_ft

W_M, D_M = 200 * FT, 100 * FT


def google_response(margin_m=0.0, pitch=2.0):
    """Panels tiled edge to edge over the 200 x 100 ft roof, like an unconstrained layout."""
    ph, pw = 1.879, 1.045  # Google's default panel
    panels = []
    y = -D_M / 2 + margin_m + pw / 2
    while y + pw / 2 <= D_M / 2 - margin_m:
        x = -W_M / 2 + margin_m + ph / 2
        while x + ph / 2 <= W_M / 2 - margin_m:
            lon, lat = FRAME.point_to_lonlat(x, y)
            panels.append({"center": {"latitude": lat, "longitude": lon}, "orientation": "LANDSCAPE", "segmentIndex": 0})
            x += ph
        y += pw
    return {
        "imageryQuality": "HIGH",
        "imageryDate": {"year": 2025, "month": 6, "day": 1},
        "solarPotential": {
            "maxArrayPanelsCount": len(panels),
            "panelCapacityWatts": 400,
            "panelHeightMeters": ph,
            "panelWidthMeters": pw,
            "roofSegmentStats": [{"pitchDegrees": pitch, "azimuthDegrees": 180.0}],
            "solarPanels": panels,
        },
    }


def test_google_filter_drops_panels_in_fire_pathways():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    ins = GoogleInsights.from_response(google_response())
    est = GoogleFilteredEstimator(GeometricEstimator(design=DesignConfig(flat_racking=Racking.FLUSH)))
    r = est.estimate(b, ins)
    total, kept = r.details["google_panels_total"], r.details["google_panels_kept"]
    assert 0 < kept < total
    # usable area is (192 x 92 ft) minus a 4 ft x 92 ft gap; kept panels cannot exceed it
    usable = (192 * 92 - 4 * 92) * FT * FT
    assert kept * 1.879 * 1.045 <= usable
    assert r.module_count == int(kept * 1.879 * 1.045 // Module().area_m2)


def test_google_flat_density_applies_racking_gcr():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    ins = GoogleInsights.from_response(google_response())
    flush = GoogleFilteredEstimator(GeometricEstimator(design=DesignConfig(flat_racking=Racking.FLUSH))).estimate(b, ins)
    ew = GoogleFilteredEstimator(GeometricEstimator(design=DesignConfig(east_west_gcr=0.9))).estimate(b, ins)
    assert ew.module_count == pytest.approx(flush.module_count * 0.9, abs=1)


def test_pipeline_prefers_google_and_flags_disagreement():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    geo = GeometricEstimator(design=DesignConfig(flat_racking=Racking.FLUSH))
    goo = GoogleFilteredEstimator(geo)

    agree = estimate_site(b, geo, goo, GoogleInsights.from_response(google_response()))
    assert agree.method == "google_filtered"
    assert not any(r.startswith("methods_disagree") for r in agree.reasons)

    # Google sees a mostly cluttered roof: only a central patch of panels
    sparse = GoogleInsights.from_response(google_response(margin_m=12.0))
    disagree = estimate_site(b, geo, goo, sparse)
    assert any(r.startswith("methods_disagree") for r in disagree.reasons)
    assert disagree.needs_review


def test_pipeline_falls_back_to_geometric_on_low_imagery():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    geo = GeometricEstimator(design=DesignConfig(flat_racking=Racking.FLUSH))
    resp = google_response()
    resp["imageryQuality"] = "LOW"
    est = estimate_site(b, geo, GoogleFilteredEstimator(geo), GoogleInsights.from_response(resp))
    assert est.method == "geometric"
    assert est.needs_review


def test_calibration_removes_bias_out_of_sample():
    rng = random.Random(0)
    samples = []
    for _ in range(60):
        true = rng.uniform(50, 800)
        samples.append(Sample("geometric|R-2|flat", true * 1.25 * rng.uniform(0.95, 1.05), true))
    raw = accuracy([s.predicted_kw for s in samples], [s.true_kw for s in samples])
    cv = cross_validate(samples)
    assert raw["within_10pct"] == 0.0
    assert cv["within_10pct"] > 0.95
    cal = Calibrator().fit(samples)
    assert cal.factor_for("geometric|R-2|flat") == (pytest.approx(0.8, abs=0.02), "segment")
    assert cal.factor_for("unknown")[1] == "global"


def test_cli_size_calibrate_evaluate(tmp_path):
    fp = centered_box_ft(200, 100)
    feats = [
        {"type": "Feature", "geometry": json.loads(json.dumps(fp.__geo_interface__)), "properties": {"id": f"b{i}", "occupancy": "commercial"}}
        for i in range(6)
    ]
    bpath = tmp_path / "b.geojson"
    bpath.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    out = tmp_path / "r.csv"
    assert main(["size", "--buildings", str(bpath), "--out", str(out), "--layouts", str(tmp_path / "l.geojson")]) == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and float(rows[0]["dc_kw"]) > 0

    truth = tmp_path / "t.csv"
    with open(truth, "w") as f:
        f.write("building_id,true_kw\n")
        for i, r in enumerate(rows):
            f.write(f"{r['building_id']},{float(r['raw_kw']) * (0.85 + 0.01 * i)}\n")
    cal = tmp_path / "cal.json"
    assert main(["calibrate", "--results", str(out), "--truth", str(truth), "--out", str(cal)]) == 0
    assert main(["size", "--buildings", str(bpath), "--out", str(out), "--calibration", str(cal)]) == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["calibration_source"] == "segment"
    assert main(["evaluate", "--results", str(out), "--truth", str(truth)]) == 0


def test_google_client_caches_not_found(tmp_path, monkeypatch):
    import requests
    from rooftop_solar.sources.google_solar import GoogleSolarClient

    calls = []

    def fake_get(self, url, params=None, timeout=None):
        calls.append(params)
        r = requests.Response()
        r.status_code = 404
        r.url = url
        return r

    monkeypatch.setattr(requests.Session, "get", fake_get)
    client = GoogleSolarClient("k", cache_dir=tmp_path)
    for _ in range(2):
        with pytest.raises(requests.HTTPError) as exc:
            client.building_insights(34.0, -118.0)
        assert exc.value.response.status_code == 404
    assert len(calls) == 1


def test_north_facing_pitched_panels_left_out_by_default():
    b = Building("a", centered_box_ft(200, 100), Occupancy.R2)
    north = GoogleInsights.from_response(google_response(pitch=25.0))
    north.segments[0].azimuth_deg = 0.0  # every panel on a north-facing slope
    south = GoogleInsights.from_response(google_response(pitch=25.0))
    geo = GeometricEstimator()
    r_n = GoogleFilteredEstimator(geo).estimate(b, north)
    r_s = GoogleFilteredEstimator(geo).estimate(b, south)
    assert r_n.module_count == 0 and r_n.details["poleward_face_kw"] > 0
    assert r_s.module_count > 0 and r_s.details["poleward_face_kw"] == 0
    keep = GoogleFilteredEstimator(GeometricEstimator(design=DesignConfig(exclude_poleward_faces=False))).estimate(b, north)
    assert keep.module_count > 0


def test_low_yield_panels_dropped_by_energy_cutoff():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    resp = google_response()
    for i, p in enumerate(resp["solarPotential"]["solarPanels"]):
        p["yearlyEnergyDcKwh"] = 600.0 if i % 2 == 0 else 300.0  # half the panels produce half as much
    ins = GoogleInsights.from_response(resp)
    flush = DesignConfig(flat_racking=Racking.FLUSH, min_panel_energy_ratio=0.0)
    all_kept = GoogleFilteredEstimator(GeometricEstimator(design=flush)).estimate(b, ins)
    import dataclasses
    cut = GoogleFilteredEstimator(GeometricEstimator(design=dataclasses.replace(flush, min_panel_energy_ratio=0.8))).estimate(b, ins)
    assert cut.module_count == pytest.approx(all_kept.module_count / 2, abs=2)
    assert cut.details["low_yield_kw"] > 0


def test_energy_cutoff_uses_typical_panel_not_a_few_outliers():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    resp = google_response()
    panels = resp["solarPotential"]["solarPanels"]
    for i, p in enumerate(panels):
        p["yearlyEnergyDcKwh"] = 900.0 if i < 3 else 600.0  # three sunny outliers, the rest uniform
    import dataclasses
    flush = DesignConfig(flat_racking=Racking.FLUSH)
    ins = GoogleInsights.from_response(resp)
    base = GoogleFilteredEstimator(GeometricEstimator(design=flush)).estimate(b, ins)
    cut = GoogleFilteredEstimator(GeometricEstimator(design=dataclasses.replace(flush, min_panel_energy_ratio=0.9))).estimate(b, ins)
    assert cut.module_count == base.module_count  # uniform roof kept despite the outliers


def test_raised_racking_fills_small_equipment_gaps_not_large_openings():
    from shapely.geometry import Point as P

    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    flush = DesignConfig(flat_racking=Racking.FLUSH, min_panel_energy_ratio=0.0)

    def with_hole(radius_m):
        resp = google_response()
        keep = []
        for p in resp["solarPotential"]["solarPanels"]:
            x, y = FRAME.point_to_local(p["center"]["longitude"], p["center"]["latitude"])
            if P(x - 15, y).distance(P(0, 0)) > radius_m:  # hole centred 15 m west of middle
                keep.append(p)
        resp["solarPotential"]["solarPanels"] = keep
        return GoogleFilteredEstimator(GeometricEstimator(design=flush)).estimate(b, GoogleInsights.from_response(resp))

    small = with_hole(2.0)   # ~4 m HVAC-sized gap
    large = with_hole(9.0)   # ~18 m opening, e.g. a courtyard
    assert small.details["raised_racking_extra_modules"] > 0
    lost_small = with_hole(0.0).module_count - small.module_count
    assert small.details["raised_racking_extra_modules"] >= 0.6 * lost_small
    assert large.details["raised_racking_extra_modules"] < 0.2 * (with_hole(0.0).module_count - large.module_count)


def test_raised_racking_fills_equipment_notch_open_to_one_side():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    flush = DesignConfig(flat_racking=Racking.FLUSH, min_panel_energy_ratio=0.0)

    def with_notch(width_m):
        resp = google_response()
        resp["solarPotential"]["solarPanels"] = [
            p for p in resp["solarPotential"]["solarPanels"]
            if not (abs(FRAME.point_to_local(p["center"]["longitude"], p["center"]["latitude"])[0] + 15) < width_m / 2
                    and FRAME.point_to_local(p["center"]["longitude"], p["center"]["latitude"])[1] > D_M / 2 - 6)
        ]
        return GoogleFilteredEstimator(GeometricEstimator(design=flush)).estimate(b, GoogleInsights.from_response(resp))

    full = with_notch(0.0)
    notch = with_notch(5.0)  # equipment cluster against the north parapet
    assert full.details["raised_racking_extra_modules"] == 0  # full roof: no gaps, no edge strips
    lost = full.module_count - notch.module_count
    assert 0.6 * lost <= notch.details["raised_racking_extra_modules"] <= 1.1 * lost


def test_raised_racking_does_not_fill_shaded_spots():
    from shapely.geometry import Point as P

    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    resp = google_response()
    for p in resp["solarPotential"]["solarPanels"]:
        x, y = FRAME.point_to_local(p["center"]["longitude"], p["center"]["latitude"])
        p["yearlyEnergyDcKwh"] = 300.0 if P(x - 15, y).distance(P(0, 0)) <= 2.0 else 600.0  # shaded patch
    design = DesignConfig(flat_racking=Racking.FLUSH, min_panel_energy_ratio=0.7)
    r = GoogleFilteredEstimator(GeometricEstimator(design=design)).estimate(b, GoogleInsights.from_response(resp))
    assert r.details["low_yield_kw"] > 0
    assert r.details["raised_racking_extra_modules"] == 0 and r.raised_areas == []


def test_kml_draws_raised_racking_areas(tmp_path):
    from rooftop_solar.sources.kml_io import write_kml_layouts
    from shapely.geometry import box

    r = GeometricEstimator().estimate(Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL))
    r.raised_areas = [box(-118.0, 34.0, -117.9999, 34.0001)]
    path = tmp_path / "l.kml"
    write_kml_layouts([r], path)
    assert path.read_text().count("raised racking only") == 1


def test_one_bad_building_does_not_stop_the_batch(monkeypatch):
    from rooftop_solar.pipeline import ReviewPolicy, _size_one

    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)

    def boom(self, building, insights):
        raise ValueError("bad geometry")

    monkeypatch.setattr(GoogleFilteredEstimator, "estimate", boom)
    est = _size_one((b, GeometricEstimator(), GoogleInsights.from_response(google_response()), None, None, ReviewPolicy()))
    assert est.method == "geometric" and est.dc_kw > 0
    assert est.reasons[0] == "google_sizing_error:ValueError_sized_from_outline" and est.needs_review


def test_pitched_layout_kept_inside_plane_edge_setback():
    from rooftop_solar.fire_code import FireCodeRules

    b = Building("a", centered_box_ft(200, 100), Occupancy.R2)
    resp = google_response(pitch=25.0)
    # west plane (segment 0) and east plane (segment 1) meet at a ridge along x = 15 m (off the
    # 4 ft array-section pathway at x = 0, which would already clear the ridge)
    resp["solarPotential"]["roofSegmentStats"] = [{"pitchDegrees": 25.0, "azimuthDegrees": 270.0},
                                                  {"pitchDegrees": 25.0, "azimuthDegrees": 90.0}]
    for p in resp["solarPotential"]["solarPanels"]:
        x, _y = FRAME.point_to_local(p["center"]["longitude"], p["center"]["latitude"])
        p["segmentIndex"] = 0 if x < 15 else 1
        p["orientation"] = "PORTRAIT"  # long side downslope (east-west), matching the grid spacing
    ins = GoogleInsights.from_response(resp)
    design = DesignConfig(min_panel_energy_ratio=0.0)
    count = lambda inches: GoogleFilteredEstimator(GeometricEstimator(FireCodeRules(residential_setback_ft=inches / 12),
                                                                      design)).estimate(b, ins)
    none, r18, r36 = count(0), count(18), count(36)
    assert none.module_count > r18.module_count >= r36.module_count > 0  # panels pulled back from the ridge
    assert r36.details["pitched_setback_kw"] > 0 and none.details["pitched_setback_kw"] == 0


def test_pitched_r2_uses_plane_setback_not_commercial_perimeter():
    from rooftop_solar.fire_code import FireCodeRules

    resp = google_response(pitch=25.0)
    ins = GoogleInsights.from_response(resp)
    design = DesignConfig(min_panel_energy_ratio=0.0)
    rules = FireCodeRules(residential_setback_ft=0)
    r2 = GoogleFilteredEstimator(GeometricEstimator(rules, design)).estimate(Building("a", centered_box_ft(200, 100), Occupancy.R2), ins)
    com = GoogleFilteredEstimator(GeometricEstimator(rules, design)).estimate(
        Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL), ins)
    assert r2.module_count > com.module_count  # no 4 ft perimeter on top of the plane setback for R-2


def test_census_result_cached_under_google_is_retried(tmp_path, monkeypatch):
    import json

    from rooftop_solar.sources import geocode as geocode_mod
    from rooftop_solar.sources.geocode import Geocoder

    cache = tmp_path / "g.json"
    cache.write_text(json.dumps({"google|1 main st": {"lat": 1, "lon": 2, "source": "census", "precision": "interpolated",
                                                      "matched_address": ""}}))

    class R:
        status_code = 200

        def json(self):
            return {"status": "OK", "results": [{"formatted_address": "1 Main St", "geometry": {
                "location": {"lat": 34.2, "lng": -118.2}, "location_type": "ROOFTOP"}}]}

    monkeypatch.setattr(geocode_mod.requests.Session, "get", lambda self, url, params=None, timeout=None: R())
    r = Geocoder("google", api_key="k", cache_path=cache).geocode("1 Main St")
    assert r.source == "google" and r.lat == 34.2


def test_pitched_setback_trims_only_the_ridge_row():
    from rooftop_solar.fire_code import FireCodeRules

    b = Building("a", centered_box_ft(200, 100), Occupancy.R2)
    ins = GoogleInsights.from_response(google_response(pitch=25.0))  # one south-facing plane, ridge on the north
    design = DesignConfig(min_panel_energy_ratio=0.0)
    est = lambda inches: GoogleFilteredEstimator(GeometricEstimator(FireCodeRules(residential_setback_ft=inches / 12),
                                                                    design)).estimate(b, ins)
    none, r18 = est(0), est(18)
    lost = none.details["google_panels_kept"] - r18.details["google_panels_kept"]
    per_row = sum(1 for p in ins.panels if abs(p.lat - max(q.lat for q in ins.panels)) < 1e-7)
    assert 0 < lost <= per_row  # the top (ridge) row at most; eave and side rows stay
