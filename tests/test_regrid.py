import csv

import pytest
from shapely.geometry import box, mapping

from rooftop_solar import GeometricEstimator
from rooftop_solar.sites import Site, size_sites
from rooftop_solar.sources import regrid as regrid_mod
from rooftop_solar.sources.regrid import RegridClient, RegridError

from .helpers import FRAME, ll


def _v2_response(*polys_local, apn="123-456"):
    feats = []
    for i, p in enumerate(polys_local):
        feats.append({
            "type": "Feature",
            "geometry": mapping(FRAME.to_lonlat(p)),
            "properties": {"headline": f"{i} Main St", "fields": {"parcelnumb": f"{apn}-{i}", "address": f"{i} Main St",
                                                                  "owner": "Owner LLC", "ll_gisacre": 2.5}},
        })
    return {"parcels": {"type": "FeatureCollection", "features": feats}}


class _Resp:
    def __init__(self, status, data=None):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


def test_parcel_at_picks_containing_parcel_and_caches(tmp_path, monkeypatch):
    calls = []
    data = _v2_response(box(-50, -50, -10, 50), box(0, -50, 60, 50))

    def fake_get(self, url, params=None, timeout=None):
        calls.append(params)
        return _Resp(200, data)

    monkeypatch.setattr(regrid_mod.requests.Session, "get", fake_get)
    client = RegridClient("t", cache_dir=tmp_path)
    lon, lat = ll(30, 0)
    p = client.parcel_at(lat, lon)
    assert p.parcel_id == "123-456-1" and p.acres == 2.5
    client.parcel_at(lat, lon)
    assert len(calls) == 1  # second lookup served from cache


def test_parcel_errors_are_reason_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(regrid_mod.requests.Session, "get", lambda self, url, params=None, timeout=None: _Resp(403))
    with pytest.raises(RegridError, match="http_403"):
        RegridClient("t", cache_dir=tmp_path).parcel_at(34.0, -118.0)
    monkeypatch.setattr(regrid_mod.requests.Session, "get",
                        lambda self, url, params=None, timeout=None: _Resp(200, {"parcels": {"features": []}}))
    with pytest.raises(RegridError, match="no_parcel_at_point"):
        RegridClient("t", cache_dir=tmp_path / "b").parcel_at(34.0, -118.0)


class _FakeParcels:
    """Parcel around both apartment buildings of the test dataset (apt-a, apt-b), not the others."""

    def parcel_at(self, lat, lon):
        f = _v2_response(box(190, -10, 250, 55))["parcels"]["features"][0]
        return regrid_mod._parcel(f)


def test_parcel_brings_in_every_building_on_it(footprints):
    lon, lat = ll(220, 7)
    out = size_sites([Site("Oak", lat=lat, lon=lon)], footprints, GeometricEstimator(), workers=1,
                     progress=lambda *_: None, parcels=_FakeParcels())[0]
    assert sorted(m.overture_id for m in out.matches) == ["apt-a", "apt-b"]
    assert any(r.startswith("parcel_123-456-0_2_buildings") for r in out.reasons)
    row = out.row()
    assert row["parcel_id"] == "123-456-0" and row["buildings"] == 2


def test_parcel_failure_falls_back_to_address_match(footprints):
    class Broken:
        def parcel_at(self, lat, lon):
            raise RegridError("http_403")

    lon, lat = ll(220, 7)
    out = size_sites([Site("Oak", lat=lat, lon=lon)], footprints, GeometricEstimator(), workers=1,
                     progress=lambda *_: None, parcels=Broken())[0]
    assert [m.overture_id for m in out.matches] == ["apt-a"]
    assert "parcel_lookup_failed:http_403" in out.reasons


def test_saved_regrid_token_is_used_by_cli(tmp_path, footprints, monkeypatch):
    import rooftop_solar.cli as cli

    seen = {}

    class Client:
        def __init__(self, token):
            seen["token"] = token

        def parcel_at(self, lat, lon):
            return _FakeParcels().parcel_at(lat, lon)

    monkeypatch.setattr(cli, "OvertureFootprints", lambda **kw: footprints)
    monkeypatch.setattr(regrid_mod, "RegridClient", Client)
    token_file = tmp_path / "tok"
    token_file.write_text("abc\n")
    lon, lat = ll(220, 7)
    sites = tmp_path / "s.csv"
    sites.write_text(f"Name,Latitude,Longitude\nOak,{lat},{lon}\n")
    out = tmp_path / "r.csv"
    assert cli.main(["size-sites", "--sites", str(sites), "--out", str(out), "--regrid-token-file", str(token_file), "--workers", "1"]) == 0
    assert seen["token"] == "abc"
    with out.open() as f:
        row = next(csv.DictReader(f))
    assert row["parcel_id"] == "123-456-0" and row["buildings"] == "2"


def test_parcel_lookup_retries_after_timeout(tmp_path, monkeypatch):
    calls = []
    data = _v2_response(box(0, -50, 60, 50))

    def flaky(self, url, params=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise regrid_mod.requests.Timeout()
        return _Resp(200, data)

    monkeypatch.setattr(regrid_mod.requests.Session, "get", flaky)
    monkeypatch.setattr(regrid_mod.time, "sleep", lambda s: None)
    lon, lat = ll(30, 0)
    assert RegridClient("t", cache_dir=tmp_path).parcel_at(lat, lon).parcel_id == "123-456-0"
    assert len(calls) == 2

    def always_slow(self, url, params=None, timeout=None):
        raise regrid_mod.requests.Timeout()

    monkeypatch.setattr(regrid_mod.requests.Session, "get", always_slow)
    with pytest.raises(RegridError, match="timeout"):
        RegridClient("t", cache_dir=tmp_path / "x").parcel_at(lat + 1, lon)


def test_structure_kind_flags_carports_and_garages():
    from rooftop_solar.sites import structure_kind
    from rooftop_solar.sources.overture import FootprintMatch

    def m(b, cls=None):
        return FootprintMatch("x", FRAME.to_lonlat(b), 0.0, cls, None, None, None, None)

    assert structure_kind(m(box(0, 0, 40, 15))) == "building"
    assert structure_kind(m(box(0, 0, 60, 6))) == "carport_or_garage"  # long narrow carport row
    assert structure_kind(m(box(0, 0, 8, 8))) == "carport_or_garage"  # 64 m2 shed
    assert structure_kind(m(box(0, 0, 40, 15), "garage")) == "carport_or_garage"


def test_outline_only_buildings_not_counted_when_google_covers_parcel(footprints):
    from rooftop_solar.sources.google_solar import GoogleSolarClient
    from .test_accuracy import _fake_insights

    class Client(GoogleSolarClient):
        """Google knows apt-a only; apt-b gets outline-only sizing."""

        def __init__(self):
            pass

        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            x, y = FRAME.point_to_local(lon, lat)
            if y > 25:
                resp = regrid_mod.requests.Response()
                resp.status_code = 404
                raise regrid_mod.requests.HTTPError(response=resp)
            clon, clat = FRAME.point_to_lonlat(220, 7)
            return _fake_insights(clat, clon)

    lon, lat = ll(220, 7)
    out = size_sites([Site("Oak", lat=lat, lon=lon)], footprints, GeometricEstimator(), google_client=Client(),
                     workers=1, progress=lambda *_: None, parcels=_FakeParcels())[0]
    assert len(out.estimates) == 2 and out.counted.count(True) == 1
    assert "not_counted_1_buildings_without_google_data" in out.reasons
    row = out.row()
    assert row["dc_kw"] == pytest.approx(sum(e.dc_kw for e, c in zip(out.estimates, out.counted) if c), abs=0.01)
    assert row["uncounted_buildings_kw"] > 0


def _dataset_with_carports(tmp_path):
    import pyarrow as pa
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    from rooftop_solar.sources.overture import OvertureFootprints

    shapes = [("apt", box(0, 0, 40, 15)), ("carport-sun", box(0, 30, 60, 36)), ("carport-shade", box(0, 50, 60, 56))]
    rows = {k: [] for k in ("id", "geometry", "bbox", "height", "num_floors", "class", "subtype", "roof_shape")}
    for bid, b in shapes:
        g = FRAME.to_lonlat(b)
        x0, y0, x1, y1 = g.bounds
        for k, v in (("id", bid), ("geometry", g.wkb), ("bbox", {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1}),
                     ("height", None), ("num_floors", None), ("class", None), ("subtype", None), ("roof_shape", None)):
            rows[k].append(v)
    d = tmp_path / "b"
    d.mkdir()
    pq.write_table(pa.table(rows), d / "part-0.parquet")
    return OvertureFootprints(cache_dir=tmp_path / "c", release="t", filesystem=pafs.LocalFileSystem(), base_path=str(d))


def test_unshaded_carports_counted_separately_shaded_ones_not(tmp_path):
    from rooftop_solar.sources.google_solar import GoogleSolarClient

    fp = _dataset_with_carports(tmp_path)

    class Parcels:
        def parcel_at(self, lat, lon):
            return regrid_mod._parcel(_v2_response(box(-5, -5, 65, 60))["parcels"]["features"][0])

    class Client(GoogleSolarClient):
        """A row of flat panels along whichever structure the point falls on."""

        def __init__(self):
            pass

        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            x, y = FRAME.point_to_local(lon, lat)
            name, y0, kwh = ("apt", 7.5, 600.0) if y < 20 else ("sun", 33.0, 580.0) if y < 45 else ("shade", 53.0, 300.0)
            panels = []
            for i in range(14):
                plon, plat = FRAME.point_to_lonlat(4 + i * 2.0, y0)
                panels.append({"center": {"latitude": plat, "longitude": plon}, "orientation": "LANDSCAPE",
                               "segmentIndex": 0, "yearlyEnergyDcKwh": kwh})
            clon, clat = FRAME.point_to_lonlat(18, y0)
            return {"name": f"buildings/{name}", "center": {"latitude": clat, "longitude": clon}, "imageryQuality": "HIGH",
                    "imageryDate": {"year": 2025, "month": 1, "day": 1},
                    "solarPotential": {"maxArrayPanelsCount": 14, "panelCapacityWatts": 400, "panelHeightMeters": 1.879,
                                       "panelWidthMeters": 1.045, "roofSegmentStats": [{"pitchDegrees": 1, "azimuthDegrees": 180}],
                                       "solarPanels": panels}}

    from rooftop_solar import DesignConfig, Racking

    lon, lat = ll(20, 7)
    out = size_sites([Site("Oak", lat=lat, lon=lon)], fp, GeometricEstimator(design=DesignConfig(flat_racking=Racking.FLUSH)),
                     google_client=Client(), workers=1, progress=lambda *_: None, parcels=Parcels(), carport_min_kw=0)[0]
    assert sorted(b.structure for b in out.buildings) == ["building", "carport_or_garage", "carport_or_garage"]
    row = out.row()
    assert row["rooftop_kw"] > 0 and row["carport_kw"] > 0
    assert "not_counted_1_shaded_carports" in out.reasons
    assert row["dc_kw"] == pytest.approx(row["rooftop_kw"] + row["carport_kw"], abs=0.05)
    assert len(out.rooftop_estimates()) == 1
    small = size_sites([Site("Oak", lat=lat, lon=lon)], fp, GeometricEstimator(design=DesignConfig(flat_racking=Racking.FLUSH)),
                       google_client=Client(), workers=1, progress=lambda *_: None, parcels=Parcels(), carport_min_kw=1e6)[0]
    assert any(r.startswith("not_counted_1_carports_or_garages_under_") for r in small.reasons)


def test_outline_only_main_building_counted_scaled_and_flagged(footprints):
    """Google knows only a small building; the big one (most of the roof) is sized from its outline."""
    from rooftop_solar.sources.google_solar import GoogleSolarClient
    from .test_accuracy import _fake_insights

    class Parcels:
        def parcel_at(self, lat, lon):
            return regrid_mod._parcel(_v2_response(box(-10, -10, 250, 25))["parcels"]["features"][0])  # warehouse (62%) + apt-a

    class Client(GoogleSolarClient):
        def __init__(self):
            pass

        def building_insights(self, lat, lon, required_quality="MEDIUM"):
            x, _y = FRAME.point_to_local(lon, lat)
            if x < 100:  # nothing for the warehouse
                resp = regrid_mod.requests.Response()
                resp.status_code = 404
                raise regrid_mod.requests.HTTPError(response=resp)
            clon, clat = FRAME.point_to_lonlat(220, 7)
            return _fake_insights(clat, clon)

    lon, lat = ll(220, 7)
    out = size_sites([Site("Mall", lat=lat, lon=lon)], footprints, GeometricEstimator(), google_client=Client(),
                     workers=1, progress=lambda *_: None, parcels=Parcels())[0]
    assert out.counted == [True, True]
    wh = next(e for b, e in zip(out.buildings, out.estimates) if b.id.endswith("warehouse") or e.method == "geometric")
    assert wh.dc_kw == pytest.approx(0.55 * wh.raw_kw)
    assert "most_roof_area_sized_from_outlines" in out.manual_review


def test_same_owner_matches_entity_variants_and_mailing_address():
    from rooftop_solar.sources.regrid import Parcel, same_owner

    p = lambda owner, mail="": Parcel(box(0, 0, 1, 1), "1", "", owner, None, mail)
    assert same_owner(p("UDR Union Place, L.L.C."), p("UDR UNION PLACE LLC"))
    assert same_owner(p("Union Place Phase II LLC", "1745 Shea Center Dr 80129"), p("UDR Inc", "1745 SHEA CENTER DR 80129"))
    assert not same_owner(p("UDR Union Place LLC"), p("Franklin Self Storage LP"))
    assert not same_owner(p(""), p(""))


def test_unit_addresses_pull_in_same_owner_parcels_only(footprints):
    """Main parcel holds apt-a; unit points lead to apt-b (same owner) and 'unknown' (another owner)."""

    def parcel(poly, owner):
        f = _v2_response(poly)["parcels"]["features"][0]
        f["properties"]["fields"]["owner"] = owner
        return regrid_mod._parcel(f)

    class Parcels:
        calls = 0

        def parcel_at(self, lat, lon):
            self.calls += 1
            x, y = FRAME.point_to_local(lon, lat)
            if x > 300:
                return parcel(box(390, -10, 440, 30), "Corner Store LLC")
            if y > 25:
                return parcel(box(190, 25, 250, 55), "OAK APARTMENTS, L.L.C.")
            return parcel(box(190, -10, 250, 22), "Oak Apartments LLC")

    class Units:
        def unit_points(self, address):
            pts = [(220, 7), (215, 37), (225, 37), (415, 10)]  # two units in apt-b: one parcel lookup
            return [FRAME.point_to_lonlat(x, y)[::-1] for x, y in pts]

    lon, lat = ll(220, 7)
    parcels = Parcels()
    out = size_sites([Site("Oak", address="1 Oak St, Los Angeles, CA 90012", lat=lat, lon=lon)], footprints,
                     GeometricEstimator(), workers=1, progress=lambda *_: None, parcels=parcels, unit_addresses=Units())[0]
    assert sorted(m.overture_id for m in out.matches) == ["apt-a", "apt-b"]
    assert "added_1_buildings_from_1_more_parcels_same_owner" in out.reasons
    assert "skipped_1_parcels_other_owner_under_unit_addresses" in out.reasons
    assert parcels.calls == 3  # site parcel + apt-b + store, not one per unit


def test_zip_filled_from_parcel_and_units_counted(footprints):
    """Address without ZIP: the unit lookup gets the parcel's ZIP; units are reported."""
    seen = []

    class Units:
        def address_records(self, address):
            seen.append(address)
            lon, lat = FRAME.point_to_lonlat(220, 7)
            return [{"number": "1", "unit": u, "lat": lat, "lon": lon} for u in ("101", "102", "103")]

    class Parcels:
        def parcel_at(self, lat, lon):
            f = _v2_response(box(190, -10, 250, 22))["parcels"]["features"][0]
            f["properties"]["fields"].update(szip5="90012", numunits="48")
            return regrid_mod._parcel(f)

    lon, lat = ll(220, 7)
    out = size_sites([Site("Oak", address="1 Oak St, Los Angeles, CA", lat=lat, lon=lon)], footprints,
                     GeometricEstimator(), workers=1, progress=lambda *_: None, parcels=Parcels(), unit_addresses=Units())[0]
    assert seen == ["1 Oak St, Los Angeles, CA 90012"]
    row = out.row()
    assert row["units_in_address_data"] == 3 and row["units_in_parcel_records"] == 48
    assert "zip_90012_added_for_unit_lookup" in out.reasons


def _owned(poly, owner):
    f = _v2_response(poly)["parcels"]["features"][0]
    f["properties"]["fields"]["owner"] = owner
    return regrid_mod._parcel(f)


class _LotParcels:
    """Single-family lots: the site's lot (apt-a), a same-owner lot (apt-b), a neighbour's lot (unknown)."""

    other_owner_everywhere = False

    def __init__(self):
        self.calls = 0

    def parcel_at(self, lat, lon):
        self.calls += 1
        x, y = FRAME.point_to_local(lon, lat)
        if x > 300 or (self.other_owner_everywhere and self.calls > 1):
            return _owned(box(390, -10, 440, 30), "Jane Smith")
        if y > 25:
            return _owned(box(190, 25, 250, 55), "AHV COMMUNITIES, L.L.C.")
        return _owned(box(190, -10, 250, 22), "AHV Communities LLC")


class _Homes:
    """Address data: apt-a holds 4 townhomes, apt-b 2."""

    def address_records(self, address):
        return []

    def homes_in(self, zipcode, footprints):
        return [4 if FRAME.to_local(f).centroid.y < 20 else 2 for f in footprints]


def test_community_sizes_same_owner_sample_and_scales_to_units(footprints):
    from rooftop_solar.models import Occupancy

    lon, lat = ll(220, 7)
    parcels = _LotParcels()
    site = Site("Altura", address="1 Oak St, San Antonio, TX 78233", lat=lat, lon=lon, occupancy=Occupancy.R3, units=60)
    out = size_sites([site], footprints, GeometricEstimator(), workers=1, progress=lambda *_: None, parcels=parcels,
                     unit_addresses=_Homes())[0]
    assert sorted(m.overture_id for m in out.matches) == ["apt-a", "apt-b"]  # not the neighbour's building
    assert "community_sampled_2_buildings_within_150m" in out.reasons
    per_home = [r for r in out.reasons if r.startswith("community_60_homes_from_6_sampled_")]
    assert per_home
    kw_per_home = float(per_home[0].split("_sampled_")[1].split("_kw")[0])
    assert abs(out.dc_kw - 60 * kw_per_home) < 0.1 * out.dc_kw
    assert parcels.calls == 2  # the site's parcel + one ownership check
    assert "community_sized_from_only_2_buildings" in out.manual_review


def test_community_skips_other_owners_buildings(footprints):
    from rooftop_solar.models import Occupancy

    lon, lat = ll(220, 7)
    parcels = _LotParcels()
    parcels.other_owner_everywhere = True
    site = Site("Altura", lat=lat, lon=lon, occupancy=Occupancy.R3, units=60)
    out = size_sites([site], footprints, GeometricEstimator(), workers=1, progress=lambda *_: None, parcels=parcels)[0]
    assert [m.overture_id for m in out.matches] == ["apt-a"]
    assert "community_sampled_1_buildings_within_150m_skipped_1_other_owners" in out.reasons


def test_apartment_site_is_not_sampled(footprints):
    lon, lat = ll(220, 7)
    parcels = _LotParcels()
    out = size_sites([Site("Oak", lat=lat, lon=lon, units=200)], footprints, GeometricEstimator(), workers=1,
                     progress=lambda *_: None, parcels=parcels)[0]
    assert parcels.calls == 1 and [m.overture_id for m in out.matches] == ["apt-a"]
    assert not any(r.startswith("community_") for r in out.reasons)


def test_read_sites_reads_units_and_housing_type(tmp_path):
    from rooftop_solar.models import Occupancy
    from rooftop_solar.sites import read_sites

    p = tmp_path / "s.csv"
    p.write_text("Name,Address,City,State,Zip,Units,HousingType\n"
                 "Altura,13003 Toepperwein Road,San Antonio,TX,78233,316,Single-Family BTR\n"
                 "Tower,1 Main St,Austin,TX,78701,\"1,200\",Mid-Rise\n")
    a, b = read_sites(str(p))
    assert a.units == 316 and a.occupancy == Occupancy.R3
    assert b.units == 1200 and b.occupancy is None
