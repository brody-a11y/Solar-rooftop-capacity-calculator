import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from rooftop_solar.sources.geocode import Geocoder
from rooftop_solar.sources.overture_addresses import OvertureAddresses, _score, parse_address, street_tokens


def test_parse_address():
    assert parse_address("4520 Alvarado Canyon Rd, San Diego, CA 92120") == ("4520", "Alvarado Canyon Rd", "92120")
    assert parse_address("200 S. San Pedro Street Suite 500, Los Angeles, CA 90012-1234") == ("200", "S. San Pedro Street", "90012")
    assert parse_address("12B Main St Apt 4, Town, CA 90001, USA") == ("12B", "Main St", "90001")
    assert parse_address("Main St, Town, CA") is None


def test_street_matching_normalises_suffixes_and_directions():
    q = street_tokens("S. San Pedro Street")
    assert q == ["S", "SAN", "PEDRO", "ST"]
    assert _score(q, street_tokens("South San Pedro St")) == 1.0
    assert _score(street_tokens("N Spring St"), street_tokens("Spring Street")) == pytest.approx(2 / 3)
    assert _score(street_tokens("Alvarado Canyon Rd"), street_tokens("Alvarado Rd")) == 0.0


@pytest.fixture
def addresses(tmp_path):
    rows = [
        ("100", "MAIN ST", None, "90001", "US", 34.0000, -118.0000),
        ("140", "MAIN ST", None, "90001", "US", 34.0000, -118.0040),
        ("141", "MAIN ST", None, "90001", "US", 34.0010, -118.0041),
        ("120", "MAIN ST", "A", "90002", "US", 35.0, -119.0),  # other zip
        ("120", "OAK AVE", None, "90001", "US", 34.5, -118.5),
    ]
    cols = list(zip(*rows))
    t = pa.table({
        "number": cols[0], "street": cols[1], "unit": pa.array(cols[2], pa.string()), "postcode": cols[3],
        "country": cols[4], "bbox": [{"xmin": lon, "xmax": lon, "ymin": lat, "ymax": lat} for lat, lon in zip(cols[5], cols[6])],
    })
    d = tmp_path / "addr"
    d.mkdir()
    pq.write_table(t, d / "part-0.parquet", row_group_size=2)
    return OvertureAddresses(cache_dir=tmp_path / "cache", release="t", filesystem=pafs.LocalFileSystem(), base_path=str(d))


def test_exact_match(addresses):
    lat, lon, matched, precision = addresses.lookup("100 Main Street, Town, CA 90001")
    assert (lat, lon, precision) == (34.0, -118.0, "address_point")


def test_interpolates_between_same_side_numbers(addresses):
    lat, lon, matched, precision = addresses.lookup("120 Main St, Town, CA 90001")
    assert precision == "interpolated"
    assert lon == pytest.approx(-118.0020)  # halfway between 100 and 140; odd 141 ignored


def test_no_match_returns_none(addresses):
    assert addresses.lookup("500 Elm St, Town, CA 90001") is None
    assert addresses.lookup("100 Main St, Town, CA 99999") is None


def test_geocoder_overture_provider(addresses, tmp_path):
    g = Geocoder("overture", cache_path=tmp_path / "g.json", addresses=addresses)
    r = g.geocode("100 Main St, Town, CA 90001")
    assert (r.source, r.precision, r.lat) == ("overture", "address_point", 34.0)


def test_unit_points_expand_address_ranges(addresses):
    pts = addresses.unit_points("100-141 Main St, Town, CA 90001")
    assert len(pts) == 3  # 100, 140 and 141 Main St; not Oak Ave or the other zip
    assert len(addresses.unit_points("100 Main St, Town, CA 90001")) == 1


def test_homes_in_counts_addresses_on_each_footprint(addresses):
    from shapely.geometry import box

    row = box(-118.0045, 33.9998, -118.0035, 34.0012)  # covers 140 and 141 Main St
    house = box(-118.00005, 33.99995, -117.99995, 34.00005)  # 100 Main St
    empty = box(-117.9, 34.2, -117.899, 34.201)
    assert addresses.homes_in("90001", [row, house, empty]) == [2, 1, 0]
    assert addresses.homes_in("", [row]) == [0]
