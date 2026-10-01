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
