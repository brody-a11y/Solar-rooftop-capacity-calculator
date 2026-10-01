"""Fire-code and clearance rules for rooftop PV.

Defaults follow IFC 2021 Section 1205 (the 2022 California Fire Code Section 1205
uses the same structure). Many AHJs amend these values, so build one
`FireCodeRules` per jurisdiction rather than editing the defaults.

IFC 1205.3 (buildings other than Group R-3):
  1205.3.1 perimeter pathway 6 ft; 4 ft where either building axis is <= 250 ft.
  1205.3.2 interior pathways at intervals <= 150 ft; 4 ft pathways to standpipes,
           vent hatches and around roof access hatches.
  1205.3.3 array sections <= 150 ft x 150 ft, separated by an 8 ft pathway
           (option 2.1) or a 4 ft pathway bordering skylights/smoke vents
           (options 2.2 and 2.3).
IFC 1205.2 (Group R-3) uses 36 in pathways and 18-36 in ridge setbacks; many AHJs
accept it for pitched multifamily roofs under the 1205.3 exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon

from .geometry import principal_axes
from .models import FT, Occupancy

# Clearance around each obstruction kind, in feet. "code" values come from IFC
# 1205.3.2; "practice" values are common design defaults, not code minimums.
DEFAULT_CLEARANCES_FT = {
    "hatch": 4.0,  # code: 4 ft pathway around roof access hatches
    "standpipe": 4.0,  # code: 4 ft pathway to standpipes
    "smoke_vent": 4.0,  # code: 1205.3.3 option 2.3
    "skylight": 4.0,  # code when skylights are used as the 2.2 ventilation option
    "hvac": 3.0,  # practice: NEC 110.26 working space at the unit disconnect
    "equipment": 3.0,  # detected from Google's surface model; Ivy: walking space around mechanical equipment
    "small_equipment": 1.0,  # detected items under 0.5 m2 (vents, pipes): practice, like "vent"
    "vent": 1.0,  # practice
    "other": 1.0,  # practice
}


@dataclass(frozen=True)
class FireCodeRules:
    perimeter_ft_small: float = 4.0
    perimeter_ft_large: float = 6.0
    large_axis_threshold_ft: float = 250.0
    max_array_section_ft: float = 150.0
    section_gap_ft: float = 4.0  # use 8.0 where the AHJ requires option 2.1
    residential_setback_ft: float = 3.0  # R-3 36 in pathway, applied to every plane edge
    # Apply the R-3 rules to pitched R-2 roofs (IFC 1205.3 exception, AHJ approval).
    residential_alternative_for_pitched_r2: bool = True
    clearances_ft: dict = field(default_factory=lambda: dict(DEFAULT_CLEARANCES_FT))

    def perimeter_m(self, footprint_m: Polygon) -> float:
        """Commercial perimeter pathway width for a footprint in metres."""
        short_axis, _long_axis, _angle = principal_axes(footprint_m)
        if short_axis <= self.large_axis_threshold_ft * FT:
            return self.perimeter_ft_small * FT
        return self.perimeter_ft_large * FT

    def clearance_m(self, kind: str) -> float:
        return self.clearances_ft.get(kind, self.clearances_ft["other"]) * FT

    def uses_residential_rules(self, occupancy: Occupancy, pitched: bool) -> bool:
        if occupancy == Occupancy.R3:
            return True
        return pitched and occupancy == Occupancy.R2 and self.residential_alternative_for_pitched_r2
