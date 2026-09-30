"""Read buildings drawn in Google Earth (KML/KMZ) and write layouts back as KML.

Drawing conventions (polygon placemark names, case-insensitive):

  anything else          a roof outline. Its name becomes the building id.
                         Put "commercial" or "R-3" in the name, description or
                         folder name to set occupancy; the default is R-2
                         (multifamily).
  hvac, vent, skylight,  an obstruction on the roof it sits inside, e.g. "HVAC 2".
  hatch, standpipe,      Once any obstruction is drawn on a roof, that roof is
  smoke vent, obstruction treated as fully mapped, so draw all of them.
  plane <pitch> <dir>    a pitched roof plane inside a roof outline, e.g.
                         "plane 6/12 S" or "plane 25 180". Pitch is x/12 or
                         degrees; direction is the downslope compass direction
                         (N, NE, ... or degrees).
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from shapely.geometry import Polygon

from ..models import Building, Obstruction, Occupancy, RoofPlane, SizingResult

_OBSTRUCTION_WORDS = {
    "hvac": "hvac",
    "rtu": "hvac",
    "vent": "vent",
    "skylight": "skylight",
    "hatch": "hatch",
    "standpipe": "standpipe",
    "smoke": "smoke_vent",
    "obstruction": "other",
}
_COMPASS = {d: i * 45.0 for i, d in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}


@dataclass
class _Shape:
    name: str
    context: str  # name + description + folder names, for occupancy keywords
    polygon: Polygon


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _child(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el if _local(c.tag) == name), None)


def _text(el: ET.Element, name: str) -> str:
    c = _child(el, name)
    return (c.text or "").strip() if c is not None else ""


def _polygons(el: ET.Element) -> list[Polygon]:
    out = []
    for poly in el.iter():
        if _local(poly.tag) != "Polygon":
            continue
        outer, holes = None, []
        for boundary in poly:
            ring = next((r for r in boundary.iter() if _local(r.tag) == "coordinates"), None)
            if ring is None:
                continue
            pts = _coords(ring.text or "")
            if _local(boundary.tag) == "outerBoundaryIs":
                outer = pts
            elif _local(boundary.tag) == "innerBoundaryIs":
                holes.append(pts)
        if outer and len(outer) >= 3:
            p = Polygon(outer, [h for h in holes if len(h) >= 3])
            out.append(p if p.is_valid else p.buffer(0))
    return out


def _walk(el: ET.Element, folders: list[str], shapes: list[_Shape]) -> None:
    for child in el:
        tag = _local(child.tag)
        if tag in ("Folder", "Document"):
            _walk(child, folders + [_text(child, "name")], shapes)
        elif tag == "Placemark":
            name = _text(child, "name")
            context = " ".join([name, _text(child, "description"), *folders])
            for p in _polygons(child):
                shapes.append(_Shape(name, context, p))


def _read_kml_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as z:
            kml_name = "doc.kml" if "doc.kml" in z.namelist() else next(n for n in z.namelist() if n.lower().endswith(".kml"))
            return z.read(kml_name)
    return path.read_bytes()


def _occupancy(context: str) -> Occupancy:
    c = context.lower()
    if "commercial" in c:
        return Occupancy.COMMERCIAL
    if re.search(r"\br-?3\b", c) or "townho" in c:
        return Occupancy.R3
    return Occupancy.R2


def _parse_plane(name: str) -> tuple[float, float] | None:
    m = re.match(r"\s*plane\s+(\S+)\s+(\S+)", name, re.IGNORECASE)
    if not m:
        return None
    pitch_s, dir_s = m.group(1), m.group(2).upper()
    if "/" in pitch_s:
        rise, run = pitch_s.split("/")
        pitch = math.degrees(math.atan(float(rise) / float(run)))
    else:
        pitch = float(pitch_s.rstrip("°"))
    azimuth = _COMPASS[dir_s] if dir_s in _COMPASS else float(dir_s.rstrip("°"))
    return pitch, azimuth


def _obstruction_kind(name: str) -> str | None:
    words = re.findall(r"[a-z]+", name.lower())
    return _OBSTRUCTION_WORDS.get(words[0]) if words else None


def load_kml_buildings(path: str | Path) -> list[Building]:
    root = ET.fromstring(_read_kml_bytes(Path(path)))
    shapes: list[_Shape] = []
    _walk(root, [], shapes)

    roofs, extras = [], []
    for s in shapes:
        if _parse_plane(s.name) or _obstruction_kind(s.name):
            extras.append(s)
        else:
            roofs.append(s)

    buildings: list[Building] = []
    used: dict[str, int] = {}
    for i, s in enumerate(roofs):
        base = s.name or f"building-{i + 1}"
        used[base] = used.get(base, 0) + 1
        bid = base if used[base] == 1 else f"{base} ({used[base]})"
        buildings.append(Building(bid, s.polygon, _occupancy(s.context)))

    for s in extras:
        pt = s.polygon.representative_point()
        owner = next((b for b in buildings if b.footprint.contains(pt)), None)
        if owner is None:
            raise ValueError(f"'{s.name}' is not inside any roof outline; draw it within the roof it belongs to")
        plane = _parse_plane(s.name)
        if plane:
            owner.roof_planes.append(RoofPlane(s.polygon, *plane))
        else:
            owner.obstructions.append(Obstruction(s.polygon, _obstruction_kind(s.name)))
            owner.obstructions_mapped = True
    if not buildings:
        raise ValueError(f"no roof outlines (polygons) found in {path}")
    return buildings


def _ring(coords) -> str:
    return " ".join(f"{x:.8f},{y:.8f},0" for x, y in coords)


def write_kml_layouts(results: list[SizingResult], path: str | Path) -> None:
    """Placed modules as KML, one folder per building, for review in Google Earth."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Solar layouts</name>',
        '<Style id="module"><LineStyle><color>ff000000</color><width>0.5</width></LineStyle>'
        "<PolyStyle><color>cc8b3d1f</color></PolyStyle></Style>",
        '<Style id="roof"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle>'
        "<PolyStyle><fill>0</fill></PolyStyle></Style>",
    ]
    for r in results:
        parts.append(f"<Folder><name>{escape(r.building_id)} - {r.dc_kw:.1f} kW DC ({r.module_count} modules)</name>")
        if r.footprint is not None:
            parts.append(
                f"<Placemark><name>{escape(r.building_id)} roof</name><styleUrl>#roof</styleUrl>"
                "<Polygon><outerBoundaryIs><LinearRing>"
                f"<coordinates>{_ring(r.footprint.exterior.coords)}</coordinates>"
                "</LinearRing></outerBoundaryIs></Polygon></Placemark>"
            )
        for poly in r.layout:
            parts.append(
                "<Placemark><styleUrl>#module</styleUrl><Polygon><outerBoundaryIs><LinearRing>"
                f"<coordinates>{_ring(poly.exterior.coords)}</coordinates>"
                "</LinearRing></outerBoundaryIs></Polygon></Placemark>"
            )
        parts.append("</Folder>")
    parts.append("</Document></kml>")
    Path(path).write_text("\n".join(parts))
