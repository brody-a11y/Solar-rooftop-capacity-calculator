import csv

import pytest

from rooftop_solar import GeometricEstimator, Occupancy
from rooftop_solar.cli import main
from rooftop_solar.sites import Site, read_sites, size_sites
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
    status_code = 200
    text = ""

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


def test_google_geocoder_falls_back_to_free_sources_and_keeps_the_error(tmp_path, monkeypatch):
    def fake_get(self, url, params=None, timeout=None):
        if "census" in url:
            return _Resp({"result": {"addressMatches": [{"coordinates": {"x": -118.1, "y": 34.1}, "matchedAddress": "1 MAIN ST"}]}})
        return _Resp({"status": "REQUEST_DENIED", "error_message": "This API key is not authorized to use this service"})

    monkeypatch.setattr(geocode_mod.requests.Session, "get", fake_get)

    class NoOverture:
        def lookup(self, address):
            return None

    g = Geocoder("google", api_key="k", cache_path=tmp_path / "g.json", addresses=NoOverture())
    r = g.geocode("1 Main St, Town, CA 90001")
    assert r.source == "census" and r.lat == 34.1
    assert "REQUEST_DENIED" in g.google_errors[0] and "not authorized" in g.google_errors[0]


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
    for col in ("dc_kw_equipment_1.5ft", "dc_kw_equipment_1ft", "dc_kw_pitched_ring_6in"):  # default alternatives
        assert float(rows[0][col]) > 0
    with (tmp_path / "res_buildings.csv").open() as f:
        assert list(csv.DictReader(f))[0]["overture_class"] == "warehouse"


def test_group_column_sums_buildings_into_one_property(tmp_path, footprints, monkeypatch):
    import rooftop_solar.cli as cli

    monkeypatch.setattr(cli, "OvertureFootprints", lambda **kw: footprints)
    (la, lo), (lb, lob) = ll(220, 7)[::-1], ll(220, 37)[::-1]
    sites = tmp_path / "s.csv"
    sites.write_text(f"Name,Group,Latitude,Longitude\nApt A,Oak Complex,{la},{lo}\nApt B,Oak Complex,{lb},{lob}\n")
    out = tmp_path / "r.csv"
    assert main(["size-sites", "--sites", str(sites), "--out", str(out), "--workers", "1"]) == 0
    with (tmp_path / "r_properties.csv").open() as f:
        props = list(csv.DictReader(f))
    with out.open() as f:
        per_site = {r["site_id"]: float(r["dc_kw"]) for r in csv.DictReader(f)}
    assert [p["group"] for p in props] == ["Oak Complex"]
    assert props[0]["buildings"] == "2"
    assert float(props[0]["dc_kw"]) == pytest.approx(per_site["Apt A"] + per_site["Apt B"])


class _UnitPoints:
    """Stand-in for county address data: two unit points, one on each apartment building."""

    def unit_points(self, address):
        return [ll(220, 7)[::-1], ll(220, 37)[::-1]] if address.startswith("10 Oak") else []


def test_unit_address_points_add_the_other_buildings(footprints):
    from rooftop_solar.sites import Site

    lon, lat = ll(220, 7)
    sites = [Site("Oak", "10 Oak St, Town, CA 90001", lat, lon)]
    out = size_sites(sites, footprints, GeometricEstimator(), workers=1, progress=lambda *_: None, unit_addresses=_UnitPoints())
    assert [m.overture_id for m in out[0].matches] == ["apt-a", "apt-b"]
    assert "campus_2_buildings_from_unit_address_points" in out[0].reasons


def test_google_fallback_looks_around_a_pin_on_the_sidewalk(footprints):
    from rooftop_solar.sites import Site
    from rooftop_solar.sources.google_solar import GoogleSolarClient

    # pin at (900, 900) with no mapped building; a large Google building sits just north of it
    from .test_accuracy import _fake_insights

    class Client(GoogleSolarClient):
        def __init__(self):
            pass

        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            from .helpers import FRAME

            x, y = FRAME.point_to_local(lon, lat)
            if y < 910:  # the pin: an entry canopy with a few panels
                r = _fake_insights(lat, lon)
                r["solarPotential"]["solarPanels"] = r["solarPotential"]["solarPanels"][:4]
                return r
            clon, clat = FRAME.point_to_lonlat(900, 915)
            return _fake_insights(clat, clon)

    lon, lat = ll(900, 900)
    out = size_sites([Site("New", lat=lat, lon=lon)], footprints, GeometricEstimator(), google_client=Client(),
                     workers=1, progress=lambda *_: None)
    assert "footprint_from_google_imagery_check_date" in out[0].reasons
    assert out[0].dc_kw > 1.0


def test_imagery_concerns_flag_stale_or_sparse_google_data():
    from datetime import date

    from rooftop_solar.models import SizingResult
    from rooftop_solar.pipeline import SiteEstimate
    from rooftop_solar.sites import Site, SiteOutcome, imagery_concerns
    from .helpers import centered_box_ft
    from rooftop_solar import Building

    def outcome(google_kw, footprint_kw, imagery):
        res = lambda kw, method, d: SizingResult("b", method, kw, int(kw * 2), 1, 1, "flat", details=d)
        goo = res(google_kw, "google_filtered", {"imagery_date": imagery})
        est = SiteEstimate("b", google_kw, google_kw, "google_filtered", 1, "none", None, False,
                           primary=goo, geometric=res(footprint_kw, "geometric", {}), google=goo)
        b = Building("b", centered_box_ft(100, 100), Occupancy.R2)
        return SiteOutcome(Site("s", ""), buildings=[b], estimates=[est], counted=[True])

    today = date(2026, 10, 1)
    assert imagery_concerns(outcome(90, 100, "2023-05-01"), 6, 0.5, today) == []
    assert "imagery_may_predate_building" in imagery_concerns(outcome(16, 37, "2023-05-01"), 6, 0.5, today)[0]
    assert imagery_concerns(outcome(90, 100, "2018-03-01"), 6, 0.5, today) == ["google_imagery_2018-03-01_9_years_old"]
    assert imagery_concerns(SiteOutcome(Site("s", "")), 6, 0.5, today) == ["no_building_found"]


def test_imagery_age_uses_date_behind_most_kw_and_match_doubts_flag():
    from datetime import date

    from rooftop_solar import Building
    from rooftop_solar.models import SizingResult
    from rooftop_solar.pipeline import SiteEstimate
    from rooftop_solar.sites import Site, SiteOutcome, imagery_concerns
    from .helpers import centered_box_ft

    def est(kw, imagery):
        goo = SizingResult("b", "google_filtered", kw, int(kw * 2), 1, 1, "flat", details={"imagery_date": imagery})
        geo = SizingResult("b", "geometric", kw * 1.5, 1, 1, 1, "flat")
        return SiteEstimate("b", kw, kw, "google_filtered", 1, "none", None, False, primary=goo, geometric=geo, google=goo)

    b = Building("b", centered_box_ft(100, 100), Occupancy.R2)
    o = SiteOutcome(Site("s", ""), buildings=[b, b], estimates=[est(300, "2023-01-01"), est(10, "2009-01-01")], counted=[True, True])
    today = date(2026, 10, 1)
    assert imagery_concerns(o, 6, 0.25, today) == []  # the 2009 shed doesn't make the site stale
    o.reasons.append("two_buildings_equally_close_check_match")
    assert imagery_concerns(o, 6, 0.25, today) == ["building_match_uncertain"]


def test_our_trimming_does_not_trigger_stale_imagery_flag():
    from datetime import date

    from rooftop_solar import Building
    from rooftop_solar.models import SizingResult
    from rooftop_solar.pipeline import SiteEstimate
    from rooftop_solar.sites import Site, SiteOutcome, imagery_concerns
    from .helpers import centered_box_ft

    goo = SizingResult("b", "google_filtered", 20, 40, 1, 1, "flat",
                       details={"imagery_date": "2024-01-01", "equipment_clearance_kw": 50,
                                "google_unclipped_kw": 70})
    geo = SizingResult("b", "geometric", 100, 200, 1, 1, "flat")
    est = SiteEstimate("b", 20, 20, "google_filtered", 1, "none", None, False, primary=goo, geometric=geo, google=goo)
    o = SiteOutcome(Site("s", ""), buildings=[Building("b", centered_box_ft(100, 100), Occupancy.R2)],
                    estimates=[est], counted=[True])
    assert imagery_concerns(o, 6, 0.25, date(2026, 10, 1)) == []  # Google saw 70 of 100 before our rules


def test_redact_hides_keys_in_error_text():
    from rooftop_solar.redact import redact

    msg = "403 Forbidden for url: https://x/v1/dataLayers:get?radiusMeters=88&key=AIzaSECRET123&view=IMAGERY"
    assert "AIzaSECRET123" not in redact(msg) and "key=***" in redact(msg)
    assert redact("token=abc def") == "token=*** def"


def test_too_few_kw_per_unit_goes_to_manual_review(footprints):
    lon, lat = ll(220, 7)
    big = size_sites([Site("Oak", lat=lat, lon=lon, units=5000)], footprints, GeometricEstimator(), workers=1,
                     progress=lambda *_: None)[0]
    assert any(r.endswith("_kw_per_unit_buildings_likely_missing") for r in big.manual_review)
    assert big.row()["kw_per_unit"] < 0.1
    ok = size_sites([Site("Oak", lat=lat, lon=lon, units=40)], footprints, GeometricEstimator(), workers=1,
                    progress=lambda *_: None)[0]
    assert not any("kw_per_unit" in r for r in ok.manual_review)
