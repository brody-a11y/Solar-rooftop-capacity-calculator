"""Core data types. Geometries on these objects are in lon/lat (EPSG:4326)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shapely.geometry import Polygon

FT = 0.3048  # metres per foot


class Occupancy(str, Enum):
    R3 = "R-3"  # one/two-family dwellings and townhouses (IFC 1205.2)
    R2 = "R-2"  # multifamily (IFC 1205.3 unless the AHJ allows the R-3 alternative)
    COMMERCIAL = "commercial"


class Racking(str, Enum):
    FLUSH = "flush"
    SOUTH = "south_tilt"
    EAST_WEST = "east_west"


@dataclass(frozen=True)
class Module:
    """PV module used for the final count. Defaults are a generic 144-half-cell module."""

    length_m: float = 2.278
    width_m: float = 1.134
    watts_dc: float = 550.0

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def watts_per_m2(self) -> float:
        return self.watts_dc / self.area_m2


@dataclass
class Obstruction:
    geometry: Polygon
    kind: str = "other"  # hvac, vent, skylight, hatch, standpipe, smoke_vent, other
    height_m: float = 0.0


@dataclass
class RoofPlane:
    geometry: Polygon  # plan-view outline
    pitch_deg: float = 0.0
    azimuth_deg: float = 180.0  # downslope direction, compass degrees


@dataclass
class Building:
    id: str
    footprint: Polygon
    occupancy: Occupancy = Occupancy.R2
    roof_planes: list[RoofPlane] = field(default_factory=list)
    obstructions: list[Obstruction] = field(default_factory=list)
    # True only when `obstructions` is a complete survey of the roof. When False the
    # geometric estimate is an upper bound until calibrated.
    obstructions_mapped: bool = False
    parapet_height_m: float = 0.0


@dataclass
class SizingResult:
    building_id: str
    method: str
    dc_kw: float
    module_count: int
    gross_roof_area_m2: float
    usable_area_m2: float
    roof_type: str  # flat, pitched, mixed
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    layout: list[Polygon] = field(default_factory=list)  # placed modules, lon/lat
