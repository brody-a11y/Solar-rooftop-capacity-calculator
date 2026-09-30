import math

import pytest
from shapely.geometry import box

from rooftop_solar import Building, DesignConfig, GeometricEstimator, Module, Obstruction, Occupancy, Racking, RoofPlane
from rooftop_solar.geometry import section_gaps
from rooftop_solar.layout import RowSpec, pack_rows, south_row_pitch_m, winter_profile_angle_deg
from rooftop_solar.models import FT

from .helpers import centered_box_ft, lonlat_box

FLUSH = DesignConfig(flat_racking=Racking.FLUSH)


def test_perimeter_4ft_when_an_axis_is_250ft_or_less():
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    r = GeometricEstimator(design=FLUSH).estimate(b)
    assert r.details["perimeter_m"] == pytest.approx(4 * FT, abs=1e-3)


def test_perimeter_6ft_when_both_axes_exceed_250ft():
    b = Building("a", centered_box_ft(400, 300), Occupancy.COMMERCIAL)
    r = GeometricEstimator(design=FLUSH).estimate(b)
    assert r.details["perimeter_m"] == pytest.approx(6 * FT, abs=1e-3)


def test_flush_flat_roof_matches_hand_count():
    # 200 x 100 ft roof, 4 ft perimeter -> 192 x 92 ft usable, split by one 4 ft gap
    # into two 94 ft sections. Landscape module 2.278 x 1.134 m, 0.02 m gaps:
    #   per section along row: floor((94 ft + 0.02) / 2.298) = 12, so 24 per row
    #   rows: floor((92 ft - 1.134) / 1.154) + 1 = 24
    b = Building("a", centered_box_ft(200, 100), Occupancy.COMMERCIAL)
    r = GeometricEstimator(design=FLUSH).estimate(b)
    assert r.module_count == 576
    assert r.dc_kw == pytest.approx(576 * 0.55)


def test_obstruction_with_clearance_removes_modules():
    fp = centered_box_ft(200, 100)
    base = GeometricEstimator(design=FLUSH).estimate(Building("a", fp, Occupancy.COMMERCIAL))
    hatch = Obstruction(lonlat_box(9, -1, 11, 1), kind="hatch")  # clear of the central pathway
    with_hatch = GeometricEstimator(design=FLUSH).estimate(
        Building("a", fp, Occupancy.COMMERCIAL, obstructions=[hatch], obstructions_mapped=True)
    )
    # keep-out is 2 m + 2 x 4 ft = 4.44 m square, so at least ~ (4.44^2 / 2.6) modules lost
    assert with_hatch.module_count <= base.module_count - 7
    assert "obstructions_not_mapped_upper_bound" not in with_hatch.flags


def test_east_west_density_close_to_gcr():
    b = Building("a", centered_box_ft(240, 140), Occupancy.COMMERCIAL)
    r = GeometricEstimator(design=DesignConfig(flat_racking=Racking.EAST_WEST, east_west_gcr=0.9)).estimate(b)
    density = r.module_count * Module().area_m2 / r.usable_area_m2
    assert 0.80 < density <= 0.90


def test_south_tilt_is_less_dense_than_east_west():
    b = Building("a", centered_box_ft(240, 140), Occupancy.COMMERCIAL)
    ew = GeometricEstimator(design=DesignConfig(flat_racking=Racking.EAST_WEST)).estimate(b)
    south = GeometricEstimator(design=DesignConfig(flat_racking=Racking.SOUTH)).estimate(b)
    assert 0 < south.module_count < ew.module_count


def test_pitched_r2_plane_uses_residential_setback_and_slope():
    # one 20 m x 10 m (plan) plane sloping south at 30 degrees
    plane = lonlat_box(-10, -5, 10, 5)
    b = Building("a", plane, Occupancy.R2, roof_planes=[RoofPlane(plane, 30.0, 180.0)])
    r = GeometricEstimator().estimate(b)
    # usable plan 20 - 2*0.9144 wide, 10 - 2*0.9144 deep; slope length = plan / cos(30)
    along = 20 - 2 * 0.9144
    slope = (10 - 2 * 0.9144) / math.cos(math.radians(30))
    portrait = math.floor((along + 0.02) / (1.134 + 0.02)) * math.floor((slope + 0.02) / (2.278 + 0.02))
    landscape = math.floor((along + 0.02) / (2.278 + 0.02)) * math.floor((slope + 0.02) / (1.134 + 0.02))
    assert r.module_count == max(portrait, landscape)
    assert r.roof_type == "pitched"
    assert "pitched_setbacks_approximated" in r.flags


def test_section_gaps_split_long_axis():
    ft = FT
    poly = box(0, 0, 388 * ft, 100 * ft)
    gaps = section_gaps(poly, 150 * ft, 4 * ft)
    remaining = poly.difference(gaps)
    assert len(remaining.geoms) == 3
    assert all(g.bounds[2] - g.bounds[0] <= 150 * ft + 1e-6 for g in remaining.geoms)


def test_pack_rows_rejects_modules_crossing_edge():
    ell = box(0, 0, 10, 10).difference(box(5, 5, 10, 10))  # L-shape
    placed = pack_rows(ell, RowSpec(2.0, 1.0, 1.0, 0.0))
    assert all(ell.buffer(1e-9).contains(p) for p in placed)
    assert len(placed) == 5 * 5 + 5 * 2  # five full rows of 5, five half rows of 2


def test_winter_profile_angle_reasonable():
    # LA at 10:00 solar time on the winter solstice: altitude ~26 deg, profile a bit higher
    ang = winter_profile_angle_deg(34.05)
    assert 26 < ang < 34
    pitch = south_row_pitch_m(1.134, 10.0, 34.05)
    assert 1.134 * math.cos(math.radians(10)) < pitch < 1.6
