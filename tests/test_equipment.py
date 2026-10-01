import io
import json

import numpy as np
import pytest
from shapely.geometry import box

from rooftop_solar import Building, DesignConfig, GeometricEstimator, Occupancy, Racking
from rooftop_solar.models import Obstruction
from rooftop_solar.sources.google_dsm import GoogleDSMClient, detect_equipment, mask_to_polygon, read_geotiff
from rooftop_solar.sources.google_solar import GoogleFilteredEstimator, GoogleInsights

from .helpers import FRAME, centered_box_ft
from .test_google_and_pipeline import D_M, W_M, google_response


def _roof(n=200, pixel=0.25, seed=0):
    """50 x 50 m flat roof at 20 m with 2 cm noise, three 1 m condensers and a 12 m penthouse."""
    rng = np.random.default_rng(seed)
    dsm = 20.0 + rng.normal(0, 0.02, (n, n))
    for r, c in [(40, 40), (40, 52), (120, 60)]:
        dsm[r:r + 4, c:c + 4] += 1.0
    dsm[100:148, 120:168] += 3.0  # penthouse: a roof level, not equipment
    return dsm


def test_detects_condensers_not_penthouse_or_noise():
    dsm = _roof()
    found = detect_equipment(dsm, np.ones_like(dsm, bool), 0.25)
    assert len(found) == 3
    assert all(m.sum() == 16 for m in found)


def test_roof_edge_is_not_equipment():
    dsm = np.full((120, 120), 0.0)
    dsm[20:100, 20:100] = 15.0  # building standing on the ground
    mask = np.zeros_like(dsm, bool)
    mask[20:100, 20:100] = True
    assert detect_equipment(dsm, mask, 0.25) == []


def test_mask_to_polygon_area():
    m = np.zeros((10, 10), bool)
    m[2:4, 3:7] = True
    g = mask_to_polygon(m, lambda c, r: (c * 0.25, -r * 0.25))
    assert g.area == pytest.approx(8 * 0.0625)


def _geotiff(dsm, x0, y0, pixel, epsg):
    import tifffile

    buf = io.BytesIO()
    keys = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, epsg)
    tifffile.imwrite(buf, dsm.astype("float32"), compression="zlib", extratags=[
        (33550, "d", 3, (pixel, pixel, 0.0)),
        (33922, "d", 6, (0.0, 0.0, 0.0, x0, y0, 0.0)),
        (34735, "H", len(keys), keys),
    ])
    return buf.getvalue()


def test_read_geotiff_transform():
    arr, to_crs, epsg = read_geotiff(_geotiff(np.zeros((4, 6)), 500000.0, 3760000.0, 0.25, 32611))
    assert arr.shape == (4, 6) and epsg == 32611
    assert to_crs(2, 3) == (500000.5, 3760000.0 - 0.75)


def test_client_finds_equipment_in_lonlat_and_caches(tmp_path, monkeypatch):
    from pyproj import Transformer

    fp = FRAME.to_lonlat(box(-25, -25, 25, 25))
    c = fp.centroid
    to_utm = Transformer.from_crs(4326, 32611, always_xy=True)
    cx, cy = to_utm.transform(c.x, c.y)
    tif = _geotiff(_roof(), cx - 25, cy + 25, 0.25, 32611)
    calls = []
    monkeypatch.setattr(GoogleDSMClient, "_download", lambda self, footprint: calls.append(1) or tif)
    client = GoogleDSMClient("k", cache_dir=tmp_path)
    eq = client.equipment(fp)
    assert len(eq) == 3 and calls == [1]
    first = FRAME.to_local(eq[0])
    assert first.area == pytest.approx(1.0, abs=0.05)
    assert fp.contains(eq[0])
    assert len(client.equipment(fp)) == 3 and calls == [1]  # cached


def test_panels_kept_clear_of_condenser_field_and_raised_spans_it():
    b = Building("a", centered_box_ft(200, 100), Occupancy.R2)
    # a 4 x 4 m condenser field in the middle of the west half
    field = FRAME.to_lonlat(box(-17, -2, -13, 2))
    with_eq = Building("a", b.footprint, Occupancy.R2, obstructions=[Obstruction(field, "equipment")])
    design = DesignConfig(flat_racking=Racking.FLUSH, min_panel_energy_ratio=0.0)
    est = GoogleFilteredEstimator(GeometricEstimator(design=design))
    ins = GoogleInsights.from_response(google_response())
    clear, blocked = est.estimate(b, ins), est.estimate(with_eq, ins)
    lost = clear.module_count - blocked.module_count
    # field plus 3 ft each side (5.8 m square), and every Google panel touching it:
    # about (5.8 + 1.9) x (5.8 + 1.0) = 53 m2 -> ~20 modules of 2.58 m2
    assert 12 <= lost <= 28
    assert blocked.details["equipment_detected"] == 1 and blocked.details["equipment_clearance_kw"] > 0
    assert blocked.details["raised_racking_extra_modules"] >= 0.6 * lost


def test_equipment_summary_line():
    from types import SimpleNamespace as NS

    from rooftop_solar.cli import _equipment_summary

    est = lambda s: NS(equipment_status=s)
    outs = [NS(estimates=[est("found_3_of_5"), est("failed:http_403"), est("")]), NS(estimates=[est("failed:http_403")])]
    assert _equipment_summary(outs) == ("Rooftop equipment lookups: 1 roofs checked, 3 equipment items kept of 5 detected, "
                                        "2 failed (http_403 x2)")
    assert _equipment_summary([NS(estimates=[est("")])]) == ""


def test_pipeline_records_equipment_status():
    from rooftop_solar.pipeline import estimate_many
    from rooftop_solar.sources.google_solar import GoogleSolarClient

    b = Building("a", centered_box_ft(200, 100), Occupancy.R2)

    class Client(GoogleSolarClient):
        def __init__(self):
            pass

        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            return google_response()

    class Equip:
        def equipment(self, footprint):
            return [FRAME.to_lonlat(box(-17, -2, -13, 2))]

    est, = estimate_many([b], GeometricEstimator(), Client(), workers=1, equipment_client=Equip())
    assert est.equipment_status == "found_1_of_1" and est.google.details["equipment_detected"] == 1


def test_read_geotiff_with_transformation_matrix():
    import tifffile

    buf = io.BytesIO()
    m = (0.25, 0.0, 0.0, 500000.0, 0.0, -0.25, 0.0, 3760000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    keys = (1, 1, 0, 1, 3072, 0, 1, 32611)
    tifffile.imwrite(buf, np.zeros((4, 6), "float32"), extratags=[(34264, "d", 16, m), (34735, "H", len(keys), keys)])
    arr, to_crs, epsg = read_geotiff(buf.getvalue())
    assert epsg == 32611 and to_crs(2, 3) == (500000.5, 3760000.0 - 0.75)


def test_dsm_client_retries_rate_limit(monkeypatch):
    import rooftop_solar.sources.google_dsm as gd

    class R:
        def __init__(self, code):
            self.status_code, self.content = code, b"tif"

        def raise_for_status(self):
            pass

    codes = [429, 429, 200]
    sleeps = []
    monkeypatch.setattr(gd.time, "sleep", sleeps.append)
    client = GoogleDSMClient("k", cache_dir=None)
    client.session = type("S", (), {"get": lambda self, url, params=None, timeout=None: R(codes.pop(0))})()
    assert client._get("u", {}).status_code == 200 and sleeps == [2.0, 4.0]


def test_screen_drops_walls_ridges_and_tags_vents():
    from rooftop_solar.sources.google_dsm import screen_equipment

    flat = np.array([[x, 0.0] for x in range(-10, 11)])
    pitched = np.array([[x, 40.0] for x in range(-10, 11)] * 3)
    items = [box(-1, -1, 0.5, 0.5),       # 1.5 m condenser on the flat roof
             box(3, 0, 3.6, 0.6),         # 0.36 m2 vent
             box(-10, 2, 10, 2.4),        # 20 m long, 0.4 m wide parapet/railing
             box(-8, 39, 8, 41.5),        # ridge line on the pitched roof
             box(-1, 39, 1, 41)]          # chimney among pitched panels
    kept = screen_equipment(items, flat, pitched)
    assert [k for _g, k in kept] == ["equipment", "small_equipment"]


def test_raised_racking_never_exceeds_googles_layout_with_dense_detections():
    import random

    rng = random.Random(1)
    b = Building("a", centered_box_ft(200, 100), Occupancy.R2)
    eq = [Obstruction(FRAME.to_lonlat(box(x, y, x + 1, y + 1)), "equipment")
          for x, y in [(rng.uniform(-28, 28), rng.uniform(-13, 13)) for _ in range(80)]]
    est = GoogleFilteredEstimator(GeometricEstimator(design=DesignConfig(min_panel_energy_ratio=0.0)))
    ins = GoogleInsights.from_response(google_response())
    clear = est.estimate(b, ins)
    busy = est.estimate(Building("a", b.footprint, Occupancy.R2, obstructions=eq), ins)
    assert busy.module_count < clear.module_count
    assert busy.module_count + busy.details["raised_racking_extra_modules"] <= clear.module_count * 1.05
