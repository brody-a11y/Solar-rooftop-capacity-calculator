import csv

import pytest

from rooftop_solar.cli import main
from rooftop_solar.sources import permits
from rooftop_solar.sources.permits import kw_from_text, pull_permits, rooftop_new_install


@pytest.mark.parametrize("text,kw", [
    ("INSTALL 250 KW ROOFTOP PV SYSTEM", 250.0),
    ("Install (476) 525W modules, 249.9kW DC roof mounted solar", 249.9),
    ("NEW 1.2 MW SOLAR ON WAREHOUSE ROOF", 1200.0),
    ("ROOFTOP PV 99.75 KWDC WITH 13.5 KWH BATTERY", 99.75),
    ("REROOF ONLY", None),
])
def test_kw_from_text(text, kw):
    assert kw_from_text(text) == kw


def test_rooftop_new_install_filters():
    assert rooftop_new_install("INSTALL 120 KW ROOFTOP SOLAR")
    assert not rooftop_new_install("120 KW SOLAR CARPORT IN PARKING LOT")
    assert not rooftop_new_install("REMOVE AND REINSTALL SOLAR FOR REROOF")
    assert not rooftop_new_install("NEW HVAC UNIT")


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class _Session:
    """Fake Socrata: one permit dataset, SF-style columns."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url == permits.CATALOG_URL:
            return _Resp({"results": [
                {"resource": {"id": "abcd-1234", "name": "Building Permits",
                              "columns_field_name": ["permit_number", "street_number", "street_name", "street_suffix",
                                                     "zipcode", "description", "issued_date", "location"]}},
                {"resource": {"id": "zzzz-0000", "name": "Street Trees", "columns_field_name": ["species"]}},
            ]})
        return _Resp([
            {"permit_number": "P1", "street_number": "100", "street_name": "Main", "street_suffix": "St", "zipcode": "94103",
             "description": "INSTALL 150 KW ROOFTOP SOLAR PV", "issued_date": "2022-05-01T00:00:00",
             "location": {"type": "Point", "coordinates": [-122.41, 37.77]}},
            {"permit_number": "P2", "street_number": "100", "street_name": "Main", "street_suffix": "St", "zipcode": "94103",
             "description": "REVISION TO 150 KW SOLAR", "issued_date": "2022-07-01T00:00:00"},
            {"permit_number": "P3", "street_number": "5", "street_name": "Oak", "street_suffix": "Ave", "zipcode": "94110",
             "description": "6.4 KW ROOFTOP PV (RESIDENTIAL)", "issued_date": "2023-01-01T00:00:00"},
            {"permit_number": "P4", "street_number": "9", "street_name": "Pine", "street_suffix": "St", "zipcode": "94110",
             "description": "80 KW SOLAR CARPORT", "issued_date": "2023-01-01T00:00:00"},
            {"permit_number": "P5", "street_number": "20", "street_name": "Bay", "street_suffix": "St", "zipcode": "94133",
             "description": "300 KW PV ON ROOF", "issued_date": "2012-01-01T00:00:00"},
        ])


def test_pull_permits_keeps_large_new_rooftop_systems():
    s = _Session()
    rows = pull_permits("sf", session=s)
    assert [(r["address"], r["true_kw"], r["kind"]) for r in rows] == [("100 Main St, San Francisco, CA 94103", 150.0, "floor")]
    assert rows[0]["latitude"] == 37.77 and rows[0]["longitude"] == -122.41
    data_call = s.calls[1]
    assert data_call[0] == "https://data.sfgov.org/resource/abcd-1234.json"
    assert "upper(description) like '%SOLAR%'" in data_call[1]["$where"]
    assert len(s.calls) == 2  # the tree dataset is not queried


def test_permits_command_writes_truth_csv_the_accuracy_test_reads(tmp_path, monkeypatch):
    from rooftop_solar.accuracy import load_truth

    monkeypatch.setattr(permits.requests, "Session", _Session)
    out = tmp_path / "p.csv"
    assert main(["permits", "--city", "sf", "--out", str(out)]) == 0
    truth = load_truth(out)
    (name, t), = truth.items()
    assert t["kind"] == "floor" and t["truths"][0]["kw"] == 150.0 and t["lat"] == 37.77
