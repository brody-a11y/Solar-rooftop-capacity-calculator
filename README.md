# Rooftop Solar Capacity Calculator

Estimates the maximum code-compliant DC system size for multifamily and commercial
rooftops. It runs in batch, checks its own numbers, and sends the ones it can't
trust to a person.

## How it reaches ±10%

No single data source gets there alone:

- **Footprint only**: sees the roof outline but not HVAC units, vents, skylights or
  hatches, so it overestimates cluttered commercial roofs.
- **Google Solar API**: its panel layout is built from a surface model, so it avoids
  clutter, but it ignores fire-code pathways, so it overestimates wherever setbacks
  apply.

So the system does four things:

1. **Geometric estimate.** Applies fire-code setbacks and pathways to the footprint
   (or to roof planes and obstructions, if you have them) and packs real module
   rectangles. Rows are fully inside the usable area and aligned to the building's
   main axis.
2. **Google-filtered estimate.** Takes Google's panel layout, drops every panel in
   a fire-code pathway, and rescales to your module and racking.
3. **Cross-check and review routing.** The two methods are independent. When they
   disagree beyond a tunable band, when imagery quality is low, or when only one
   method ran, the site is flagged `needs_review`.
4. **Calibration against your own designs.** Fits a correction factor per segment
   (method × occupancy × roof type) from engineered max-fill layouts, and reports
   **cross-validated** accuracy. That out-of-sample number is the one to quote.

The ±10% target applies to **auto-accepted** sites. The `evaluate` command reports
accuracy and the auto-accept rate side by side. Tighten the review band until the
accuracy target holds, then read off what share of the portfolio still needs a person.

## Fire-code rules applied (defaults)

Defaults follow IFC 2021 §1205. CFC 2022 §1205 has the same structure. AHJs amend
these rules, so set `FireCodeRules` per jurisdiction.

| Rule | Default | Source |
|---|---|---|
| Perimeter pathway | 4 ft if either axis ≤ 250 ft, else 6 ft | IFC 1205.3.1 |
| Array section max | 150 ft × 150 ft | IFC 1205.3.3 |
| Section separation | 4 ft (set 8 ft where the AHJ requires option 2.1) | IFC 1205.3.3 |
| Hatch / standpipe / smoke vent / skylight clearance | 4 ft | IFC 1205.3.2–3 |
| HVAC clearance | 3 ft | design practice (NEC 110.26 working space), not a code minimum |
| Plumbing vent / other | 1 ft | design practice |
| Pitched R-2 roofs | R-3 rules (36 in at every plane edge) | IFC 1205.3 exception, needs AHJ approval; `--no-residential-alternative` turns it off |

The pitched-roof handling is approximate. It applies 36 in at every plane edge,
which is conservative at eaves. Results carry the flag `pitched_setbacks_approximated`.

## Install and run

**Mac without Git:** follow `START HERE.txt`. `Install.command` sets it up and
`Size Buildings.command` runs it, both by double-click. The Python environment
lives in `~/.rooftop-solar`, so the folder itself can sit in Google Drive.

**Command line:**

```bash
pip install -e ".[dev]"
pytest

# 1. size a portfolio (footprints only)
rooftop-solar size --buildings buildings.geojson --out results.csv --layouts layouts.geojson

# 1b. add the Google cross-check (billed per uncached call; responses cached in .cache/)
export GOOGLE_SOLAR_API_KEY=...
rooftop-solar size --buildings buildings.geojson --out results.csv --google

# 2. calibrate against engineered designs (CSV: building_id,true_kw)
rooftop-solar calibrate --results results.csv --truth truth.csv --out calibration.json

# 3. re-size with calibration, then measure
rooftop-solar size --buildings buildings.geojson --out results.csv --google --calibration calibration.json
rooftop-solar evaluate --results results.csv --truth truth.csv
```

Set racking to what you actually spec: `--flat-racking east_west|south_tilt|flush`,
`--gcr`, `--flat-tilt-deg`, `--module-watts`, `--module-length-m`, `--module-width-m`.
The defaults (east-west at 10°, GCR 0.90, 550 W 2.278 × 1.134 m module) are
placeholders. They are not taken from a particular racking datasheet.

### Input GeoJSON

Each feature's `role` property sets what it is:

| role | geometry | properties |
|---|---|---|
| `footprint` (default) | building outline | `id`, `occupancy` (`R-2`, `R-3`, `commercial`), `obstructions_mapped` (true only if every obstruction is drawn), `parapet_height_m` |
| `roof_plane` | plan-view plane outline | `building_id`, `pitch_deg`, `azimuth_deg` (downslope, compass) |
| `obstruction` | outline | `building_id`, `kind` (`hvac`, `vent`, `skylight`, `hatch`, `standpipe`, `smoke_vent`, `other`), `height_m` |

With no roof planes, the whole footprint is treated as a flat roof.

### Output columns

`dc_kw` (calibrated), `raw_kw`, `module_count`, `method`, `geometric_kw`,
`google_kw`, `agreement_ratio`, `calibration_factor`, `needs_review`, `reasons`,
`flags`. Load `layouts.geojson` in QGIS or geojson.io to check placements by eye.

## Ground truth: what counts

Calibrate against **max-fill layouts from an engineered design tool or permit
set**. Don't use installed system sizes: those are often capped by budget, load,
interconnection or tariff limits (for example NEM/VNEM sizing), and fitting to
them teaches the model the wrong thing. Include a mix of roof types and
occupancies. Each segment needs at least 5 buildings before it gets its own
factor. Below that, the global factor is used.

## Not built yet

- **Footprint acquisition at scale.** You supply the footprints. Candidate sources
  are Overture Maps / Microsoft building footprints, OSM, or parcel data. They were
  not wired in because they couldn't be tested from this environment.
- **Obstruction detection from imagery.** This is the biggest remaining accuracy
  lever for footprint-only sites.
- **Detailed R-3 ridge/eave/rake setbacks per plane.** This needs ridge-line
  geometry.
- **Structural and wind-zone limits.** `--edge-setback-ft` covers only the simple
  case. Roof load capacity is out of scope.
- **Parallelism.** It is thread-based. A typical roof sizes in under 1 s; a
  400k sq ft roof takes about 5 s.
