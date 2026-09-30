import csv

import pytest

from rooftop_solar import GeometricEstimator, Occupancy
from rooftop_solar.cli import main
from rooftop_solar.sites import read_sites, size_sites
from rooftop_solar.sources import geocode as geocode_mod
from rooftop_solar.sources.geocode import Geocoder

from .helpers import ll


def test_point_inside_building_matches_it(footprints):
    got = footprints.find({"s": ll(30, 20)})["s"]
    assert [m.overture_id for m in got] == ["warehouse"]
    assert got[0].distance_m == 0.0
    assert (footprints.cache_dir / "index-test.json").exists()


def test_point_in_street_matches_nearest_within_search(footprints):
    got = footprints.find({"near": ll(30, -12), "far": ll(30, -80)}, search_m=40)
    assert got["near"][0].overture_id == "warehouse"
    assert got["near"][0].distance_m == pytest.approx(12, abs=0.5)
    assert got["far"] == []


def test_campus_radius_adds_neighbouring_buildings(footprints):
    one = footprints.find({"s": ll(220, 7)})["s"]
    campus = footprints.find({"s": ll(220, 7)}, campus_m=50)["s"]
    assert [m.overture_id for m in one] == ["apt-a"]
    assert [m.overture_id for m in campus] == ["apt-a", "apt-b"]


def test_read_sites_accepts_loose_headers(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(
        "Property Name,Street Address,City,State,Zip Code,Property Type\n"
        "Oak Apts,1 Main St,Los Angeles,CA,90012,Multifamily\n"
        "Big Box,2 Main St,Los Angeles,CA,90012,Commercial\n"
    )
    sites = read_sites(str(p))
    assert [s.id for s in sites] == ["Oak Apts", "Big Box"]
    assert sites[0].address == "1 Main St, Los Angeles, CA 90012"
    assert sites[0].occupancy == Occupancy.R2
    assert sites[1].occupancy == Occupancy.COMMERCIAL


def test_read_sites_does_not_repeat_city_already_in_address(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("Name,Address,City\nA,\"1 Main St, Los Angeles, CA 90012\",Los Angeles\n")
    assert read_sites(str(p))[0].address == "1 Main St, Los Angeles, CA 90012"


def test_read_sites_rejects_file_without_location_columns(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("Name,Owner\nA,B\n")
    with pytest.raises(ValueError, match="needs an 'address' column"):
        read_sites(str(p))


class _Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_census_and_google_geocoders_parse_and_cache(tmp_path, monkeypatch):
    calls = []

    def fake_get(self, url, params=None, timeout=None):
        calls.append(url)
        if "census" in url:
            return _Resp({"result": {"addressMatches": [{"coordinates": {"x": -118.1, "y": 34.1}, "matchedAddress": "1 MAIN ST"}]}})
        return _Resp({"status": "OK", "results": [{"formatted_address": "1 Main St", "geometry": {"location": {"lat": 34.2, "lng": -118.2}, "location_type": "ROOFTOP"}}]})

    monkeypatch.setattr(geocode_mod.requests.Session, "get", fake_get)
    census = Geocoder("census", cache_path=tmp_path / "c.json")
    r = census.geocode("1 Main St")
    assert (r.lat, r.lon, r.precision) == (34.1, -118.1, "interpolated")
    census.geocode("1  main st")  # same address, different spacing/case: cached
    assert len(calls) == 1
    census.save()
    assert Geocoder("census", cache_path=tmp_path / "c.json").geocode("1 Main St").lat == 34.1
    assert len(calls) == 1

    g = Geocoder("google", api_key="k", cache_path=tmp_path / "g.json").geocode("1 Main St")
    assert (g.lat, g.lon, g.precision) == (34.2, -118.2, "rooftop")


def test_ambiguous_and_far_matches_are_flagged(footprints):
    from rooftop_solar.sites import Site

    between = ll(220, 22.5)  # midway between apt-a and apt-b
    far = ll(30, -32)  # 32 m south of the warehouse
    sites = [Site("between", lat=between[1], lon=between[0]), Site("far", lat=far[1], lon=far[0])]
    out = {o.site.id: o for o in size_sites(sites, footprints, GeometricEstimator(), workers=1, progress=lambda *_: None)}
    assert "two_buildings_equally_close_check_match" in out["between"].reasons
    assert any(r.startswith("address_point_32m_from_building") for r in out["far"].reasons)


def test_size_sites_end_to_end(footprints):
    from rooftop_solar.sites import Site

    sites = [
        Site("wh", lat=ll(30, 20)[1], lon=ll(30, 20)[0]),
        Site("street", lat=ll(30, -12)[1], lon=ll(30, -12)[0]),
        Site("nothing", lat=ll(900, 900)[1], lon=ll(900, 900)[0]),
        Site("mystery", lat=ll(415, 10)[1], lon=ll(415, 10)[0]),
    ]
    out = {o.site.id: o for o in size_sites(sites, footprints, GeometricEstimator(), workers=1, progress=lambda *_: None)}
    assert out["wh"].buildings[0].occupancy == Occupancy.COMMERCIAL
    assert out["wh"].dc_kw > 0
    assert not any("check_match" in r for r in out["street"].reasons)  # 12 m off, no rival: fine
    assert any("same_building_as_site_wh" in r for r in out["street"].reasons)
    assert out["nothing"].buildings == [] and "no_building_within_40m" in out["nothing"].reasons
    assert "occupancy_assumed_multifamily" in out["mystery"].reasons
    row = out["wh"].row()
    assert row["buildings"] == 1 and row["needs_review"]


def test_cli_size_sites_with_coordinates(tmp_path, footprints, monkeypatch):
    import rooftop_solar.cli as cli

    monkeypatch.setattr(cli, "OvertureFootprints", lambda **kw: footprints)
    lon, lat = ll(30, 20)
    sites = tmp_path / "sites.csv"
    sites.write_text(f"Name,Latitude,Longitude\nWarehouse,{lat},{lon}\n")
    out = tmp_path / "res.csv"
    assert main(["size-sites", "--sites", str(sites), "--out", str(out), "--layouts", str(tmp_path / "l.kml"), "--workers", "1"]) == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["site_id"] == "Warehouse" and float(rows[0]["dc_kw"]) > 0
    with (tmp_path / "res_buildings.csv").open() as f:
        assert list(csv.DictReader(f))[0]["overture_class"] == "warehouse"
