import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from rooftop_solar import GeometricEstimator, Occupancy
from rooftop_solar.cli import main
from rooftop_solar.sources.kml_io import _parse_plane, load_kml_buildings

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_buildings.kml"


def test_sample_kml_roofs_obstructions_and_planes():
    buildings = {b.id: b for b in load_kml_buildings(SAMPLE)}
    assert set(buildings) == {"Sample Warehouse", "Sample Apartments"}
    wh, apt = buildings["Sample Warehouse"], buildings["Sample Apartments"]
    assert wh.occupancy == Occupancy.COMMERCIAL
    assert sorted(o.kind for o in wh.obstructions) == ["hatch", "hvac", "hvac"]
    assert wh.obstructions_mapped
    assert apt.occupancy == Occupancy.R2
    assert len(apt.roof_planes) == 2
    assert {p.azimuth_deg for p in apt.roof_planes} == {0.0, 180.0}
    assert GeometricEstimator().estimate(apt).roof_type == "pitched"


def test_kmz_is_read(tmp_path):
    kmz = tmp_path / "sites.kmz"
    with zipfile.ZipFile(kmz, "w") as z:
        z.writestr("doc.kml", SAMPLE.read_bytes())
    assert len(load_kml_buildings(kmz)) == 2


def test_plane_name_parsing():
    assert _parse_plane("plane 6/12 SW") == (pytest.approx(math.degrees(math.atan(0.5))), 225.0)
    assert _parse_plane("Plane 25 170") == (25.0, 170.0)
    assert _parse_plane("Building A") is None


def test_obstruction_outside_any_roof_is_an_error(tmp_path):
    kml = SAMPLE.read_text().replace("<name>Sample Warehouse</name>", "<name>HVAC stray</name>", 1)
    p = tmp_path / "bad.kml"
    p.write_text(kml)
    with pytest.raises(ValueError, match="not inside any roof outline"):
        load_kml_buildings(p)


def test_cli_writes_google_earth_layout(tmp_path):
    out, layout = tmp_path / "r.csv", tmp_path / "l.kml"
    assert main(["size", "--buildings", str(SAMPLE), "--out", str(out), "--layouts", str(layout)]) == 0
    root = ET.parse(layout).getroot()
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    folders = root.findall(".//k:Folder", ns)
    assert len(folders) == 2
    assert len(root.findall(".//k:Placemark", ns)) > 100
