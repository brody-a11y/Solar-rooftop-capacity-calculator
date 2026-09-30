"""Read buildings from GeoJSON and write layouts back out.

Each feature carries a `role` property:
  footprint   (default) properties: id, occupancy, obstructions_mapped, parapet_height_m
  roof_plane  properties: building_id, pitch_deg, azimuth_deg
  obstruction properties: building_id, kind, height_m
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import MultiPolygon, mapping, shape

from ..models import Building, Obstruction, Occupancy, RoofPlane, SizingResult


def _largest(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def load_buildings(path: str | Path) -> list[Building]:
    data = json.loads(Path(path).read_text())
    buildings: dict[str, Building] = {}
    planes, obstructions = [], []
    for i, feat in enumerate(data["features"]):
        props = feat.get("properties") or {}
        geom = _largest(shape(feat["geometry"]))
        role = props.get("role", "footprint")
        if role == "footprint":
            bid = str(props.get("id", i))
            buildings[bid] = Building(
                id=bid,
                footprint=geom,
                occupancy=Occupancy(props.get("occupancy", Occupancy.R2.value)),
                obstructions_mapped=_bool(props.get("obstructions_mapped", False)),
                parapet_height_m=float(props.get("parapet_height_m", 0.0) or 0.0),
            )
        elif role == "roof_plane":
            planes.append((str(props["building_id"]), RoofPlane(geom, float(props.get("pitch_deg", 0)), float(props.get("azimuth_deg", 180)))))
        elif role == "obstruction":
            obstructions.append(
                (str(props["building_id"]), Obstruction(geom, props.get("kind", "other"), float(props.get("height_m", 0) or 0)))
            )
        else:
            raise ValueError(f"feature {i}: unknown role {role!r}")
    for bid, plane in planes:
        buildings[bid].roof_planes.append(plane)
    for bid, ob in obstructions:
        buildings[bid].obstructions.append(ob)
    return list(buildings.values())


def write_layouts(results: list[SizingResult], path: str | Path) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {"building_id": r.building_id, "method": r.method},
        }
        for r in results
        for poly in r.layout
    ]
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": features}))
