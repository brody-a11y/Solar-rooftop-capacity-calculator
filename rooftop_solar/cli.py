"""Command line: size a portfolio, calibrate against real designs, evaluate accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from .calibration import Calibrator, Sample, accuracy, cross_validate
from .fire_code import FireCodeRules
from .models import Module, Racking
from .pipeline import ReviewPolicy, estimate_many
from .redact import redact
from .sizing import DesignConfig, GeometricEstimator
from .sources.geojson_io import load_buildings as load_geojson_buildings
from .sources.geojson_io import write_layouts as write_geojson_layouts
from .sources.kml_io import load_kml_buildings, write_kml_layouts
from .sites import group_rows, read_sites, size_sites
from .sources.geocode import Geocoder
from .sources.google_solar import GoogleSolarClient
from .sources.overture import OvertureFootprints


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_buildings(path: str):
    if path.lower().endswith((".kml", ".kmz")):
        return load_kml_buildings(path)
    return load_geojson_buildings(path)


def write_layouts(results, path: str) -> None:
    if path.lower().endswith(".kml"):
        write_kml_layouts(results, path)
    else:
        write_geojson_layouts(results, path)


def _setup(args: argparse.Namespace):
    """Estimator, calibrator and optional Google client from the shared design flags."""
    rules = FireCodeRules(
        section_gap_ft=args.section_gap_ft,
        residential_setback_ft=args.pitched_setback_in / 12.0,
        residential_alternative_for_pitched_r2=not args.no_residential_alternative,
    )
    design = DesignConfig(
        module=Module(args.module_length_m, args.module_width_m, args.module_watts),
        flat_racking=Racking(args.flat_racking),
        flat_tilt_deg=args.flat_tilt_deg,
        east_west_gcr=args.gcr,
        south_gcr=args.gcr if args.flat_racking == Racking.SOUTH.value and args.south_gcr_fixed else None,
        edge_setback_ft=args.edge_setback_ft,
        exclude_poleward_faces=not args.include_north_faces,
        min_panel_energy_ratio=args.min_panel_energy_ratio,
        min_modules_per_structure=args.min_modules_per_structure,
    )
    geometric = GeometricEstimator(rules, design)
    calibrator = Calibrator.load(args.calibration) if args.calibration else None
    client = None
    if args.google:
        key = os.environ.get(args.google_key_env)
        if not key:
            raise SystemExit(f"--google set but ${args.google_key_env} is empty")
        client = GoogleSolarClient(key, cache_dir=args.google_cache)
    return geometric, calibrator, client


def _equipment_summary(outcomes) -> str:
    """One line on the rooftop-equipment lookups, so failures show in the window."""
    from collections import Counter

    st = [e.equipment_status for o in outcomes for e in o.estimates if e.equipment_status]
    if not st:
        return ""
    ok = [s.split("_") for s in st if s.startswith("found_")]  # found_<kept>_of_<detected>
    kept, detected = sum(int(p[1]) for p in ok), sum(int(p[-1]) for p in ok)
    failed = Counter(s.split(":", 1)[1] for s in st if s.startswith("failed:"))
    line = (f"Rooftop equipment lookups: {len(ok)} roofs checked, {kept} equipment items kept of {detected} detected"
            f", {sum(failed.values())} failed")
    if failed:
        line += " (" + ", ".join(f"{k} x{v}" for k, v in failed.most_common(3)) + ")"
    return line


def _snapshots(args: argparse.Namespace, outcomes) -> None:
    """Satellite PNGs for the sites named in --snapshots (name parts, ';'-separated)."""
    wanted = [w.strip().lower() for w in (args.snapshots or "").split(";") if w.strip()]
    if not wanted:
        return
    if not args.google:
        print("Snapshots need the Google key (--google).", file=sys.stderr)
        return
    from .snapshots import snapshot
    from .sources.google_dsm import GoogleDSMClient

    client = GoogleDSMClient(os.environ[args.google_key_env])
    out_dir = args.out.rsplit(".", 1)[0] + "_snapshots"
    cache = os.path.join(os.path.dirname(os.path.abspath(args.google_cache)), "imagery_cache")
    for o in outcomes:
        if any(w in o.site.id.lower() for w in wanted):
            try:
                path = snapshot(o, client, out_dir, cache)
                print(f"Snapshot: {path}" if path else f"Snapshot: {o.site.id} has no counted buildings")
            except Exception as exc:  # one failed image must not lose the run's results
                print(f"Snapshot failed for {o.site.id}: {type(exc).__name__}: {redact(exc)}", file=sys.stderr)


def _equipment(args: argparse.Namespace):
    """Rooftop-equipment detector (Google surface model), on by default with --google."""
    if not args.google or args.no_equipment_detection:
        return None
    from .sources.google_dsm import GoogleDSMClient

    return GoogleDSMClient(os.environ[args.google_key_env])


def _write_csv(rows: list[dict], path: str, empty_header: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [empty_header])
        w.writeheader()
        w.writerows(rows)


def cmd_size(args: argparse.Namespace) -> int:
    geometric, calibrator, client = _setup(args)
    policy = ReviewPolicy()

    try:
        buildings = load_buildings(args.buildings)
    except (ValueError, KeyError) as exc:
        print(f"Couldn't read {args.buildings}: {exc}", file=sys.stderr)
        return 2

    estimates = estimate_many(buildings, geometric, client, calibrator, policy, args.workers, args.google_max_points)
    results = list(zip(buildings, estimates))

    rows = [est.row(b.occupancy.value) for b, est in results]
    _write_csv(rows, args.out, "building_id")
    if args.layouts:
        write_layouts([est.primary for _, est in results if est.primary], args.layouts)
    n_review = sum(r["needs_review"] for r in rows)
    print(f"sized {len(rows)} buildings -> {args.out}; {n_review} flagged for review")
    return 0


def _parcels(args: argparse.Namespace):
    """Regrid client when a token is available (env var or saved file), else None."""
    if args.no_parcels:
        return None
    token = os.environ.get("REGRID_TOKEN", "").strip()
    path = os.path.expanduser(args.regrid_token_file)
    if not token and os.path.exists(path):
        with open(path) as f:
            token = f.read().strip()
    if not token:
        return None
    from .sources.regrid import RegridClient

    print("Regrid parcels on (one parcel record per new site).")
    return RegridClient(token)


def _unit_addresses(args: argparse.Namespace, sites):
    if args.no_unit_points or not any(s.address for s in sites):
        return None
    from .sources.overture_addresses import OvertureAddresses

    return OvertureAddresses()


def cmd_size_sites(args: argparse.Namespace) -> int:
    geometric, calibrator, client = _setup(args)
    try:
        sites = read_sites(args.sites)
    except (ValueError, OSError) as exc:
        print(f"Couldn't read {args.sites}: {exc}", file=sys.stderr)
        return 2
    geocoder = None
    if any(s.lat is None for s in sites):
        if args.geocoder == "google":
            key = os.environ.get(args.google_key_env)
            if not key:
                print(f"--geocoder google needs ${args.google_key_env}", file=sys.stderr)
                return 2
            geocoder = Geocoder("google", key)
        else:
            geocoder = Geocoder(args.geocoder)
    footprints = OvertureFootprints(workers=args.workers)
    outcomes = size_sites(
        sites, footprints, geometric, geocoder, client, calibrator, ReviewPolicy(),
        search_m=args.search_m, campus_m=args.campus_radius_m, workers=args.workers,
        google_max_points=args.google_max_points, unit_addresses=_unit_addresses(args, sites),
        parcels=_parcels(args), include_carports=not args.no_carports, carport_min_energy_ratio=args.carport_min_energy_ratio,
        carport_min_kw=args.carport_min_kw,
        equipment_client=_equipment(args),
    )
    site_rows = [o.row() for o in outcomes]
    _write_csv(site_rows, args.out, "site_id")
    buildings_out = args.buildings_out or args.out.rsplit(".", 1)[0] + "_buildings.csv"
    _write_csv([r for o in outcomes for r in o.building_rows()], buildings_out, "site_id")
    groups = group_rows(outcomes)
    if groups:
        _write_csv(groups, args.out.rsplit(".", 1)[0] + "_properties.csv", "group")
    if args.layouts:
        write_layouts([e.primary for o in outcomes for e in o.estimates if e.primary], args.layouts)
    found = sum(1 for o in outcomes if o.buildings)
    if _equipment_summary(outcomes):
        print(_equipment_summary(outcomes))
    _snapshots(args, outcomes)
    manual = [r for r in site_rows if r["manual_review"]]
    print(f"{len(sites)} sites: {found} matched to buildings -> {args.out}")
    print(f"{len(manual)} of {len(sites)} ({len(manual) / max(len(sites), 1):.0%}) need a manual look at current imagery "
          "(manual_review column; reasons in manual_review_reason)")
    return 0


def cmd_accuracy(args: argparse.Namespace) -> int:
    import dataclasses
    import statistics

    from .accuracy import compare, load_truth, summary, truth_sites

    geometric, calibrator, client = _setup(args)
    truth = load_truth(args.truth)
    sites = truth_sites(truth)
    geocoder = None
    if any(s.lat is None for s in sites):
        key = os.environ.get(args.google_key_env) if args.geocoder == "google" else None
        geocoder = Geocoder("google", key) if key else Geocoder("auto")
    footprints, parcels, units = OvertureFootprints(workers=args.workers), _parcels(args), _unit_addresses(args, sites)
    equipment = _equipment(args)

    first = [True]

    def run(ratio: float, setback_in: float | None = None):
        # Progress on the first pass (the slow one: lookups); later passes reuse the caches.
        progress = (lambda msg: print(msg, flush=True)) if first[0] else (lambda *_: None)
        if not first[0]:
            what = f"pitched setback {setback_in:g} in" if setback_in is not None else f"panel-yield cutoff {ratio:g}"
            print(f"Re-sizing at {what}...", flush=True)
        first[0] = False
        rules = (dataclasses.replace(geometric.rules, residential_setback_ft=setback_in / 12.0)
                 if setback_in is not None else geometric.rules)
        geo = GeometricEstimator(rules, dataclasses.replace(geometric.design, min_panel_energy_ratio=ratio))
        outcomes = size_sites(
            sites, footprints, geo, geocoder, client, calibrator, ReviewPolicy(),
            search_m=args.search_m, campus_m=args.campus_radius_m, workers=args.workers,
            google_max_points=args.google_max_points, unit_addresses=units, parcels=parcels,
            include_carports=not args.no_carports, carport_min_energy_ratio=args.carport_min_energy_ratio,
            carport_min_kw=args.carport_min_kw, progress=progress,
            equipment_client=equipment,
        )
        return outcomes, compare(truth, outcomes, geo.design.module.watts_dc)

    ratios = [float(x) for x in args.energy_ratios.split(",")] if args.energy_ratios else []
    ratios = sorted(set(ratios) | {args.min_panel_energy_ratio})
    sweep = []
    for ratio in ratios:
        outcomes, rows = run(ratio)
        errs = [float(r["google_err"].rstrip("%")) / 100 for r in rows if r["google_err"]]
        sweep.append((ratio, outcomes, rows, errs))
    # Report the configured cutoff, not the best fit: reference designs are
    # cost-trimmed, so fitting them would bias absolute MaxFit low.
    ratio, outcomes, rows, _ = next(t for t in sweep if t[0] == args.min_panel_energy_ratio)
    # Pitched-roof setback sweep at the default cutoff (pitched-roof MaxFit sites only).
    setbacks = sorted({float(x) for x in args.pitched_setbacks.split(",") if x.strip()} | {args.pitched_setback_in}) \
        if args.pitched_setbacks else []
    setback_sweep = []
    for sb in setbacks:
        srows = rows if sb == args.pitched_setback_in else run(args.min_panel_energy_ratio, sb)[1]
        errs = [float(r["tool_err"].rstrip("%")) / 100 for r in srows
                if r["kind"] == "maxfit" and r["roof"] in ("pitched", "mixed") and r["tool_err"]]
        setback_sweep.append((sb, errs, {r["site"]: r["tool_err"] for r in srows if r["roof"] in ("pitched", "mixed")
                                         and r["kind"] == "maxfit"}))
    _write_csv(rows, args.out, "site")
    if args.layouts:
        write_layouts([e.primary for o in outcomes for e in o._counted() if e.primary], args.layouts)
    print()
    print(f"{'site':20} {'designs kW':17} {'standard':>13} {'raised rack':>13} {'footprint':>13}  review")
    for r in rows:
        cells = [f"{str(r[k + '_kw']):>7} {r[k + '_err']:>5}" for k in ("tool", "raised", "footprint_only")]
        designs = (">=" if r["kind"] == "floor" else "") + r["designs_kw"]
        print(f"{r['site'][:20]:20} {designs[:17]:17} {' '.join(cells)}  {'YES' if r['manual_review'] else ''}")
    print("(>= : design sized to load or budget, so MaxFit should be at least this; R : raised-racking design,\n"
          " compared with the raised column only)")
    print()
    print(summary(rows))
    if _equipment_summary(outcomes):
        print("\n" + _equipment_summary(outcomes))
    if len(sweep) > 1:
        print("\nPanel-yield cutoff sweep (Google column):")
        print(f"{'cutoff':>7} {'within 10%':>11} {'median |err|':>13} {'median err':>11}")
        for r, _o, _rows, errs in sweep:
            hits = sum(abs(e) <= 0.10 for e in errs)
            med_abs = statistics.median([abs(e) for e in errs]) if errs else float("nan")
            med = statistics.median(errs) if errs else float("nan")
            mark = "  <- shown above (default; set with --min-panel-energy-ratio)" if r == ratio else ""
            print(f"{r:7.2f} {hits:>5} of {len(errs):<3} {med_abs:>12.0%} {med:>+11.0%}{mark}")
    if len(setback_sweep) > 1:
        print("\nPitched-roof setback sweep (MaxFit sites with mostly pitched roofs, standard column):")
        print(f"{'setback':>8} {'within 10%':>11} {'median |err|':>13} {'median err':>11}  per site")
        for sb, errs, per in setback_sweep:
            hits = sum(abs(e) <= 0.10 for e in errs)
            med_abs = statistics.median([abs(e) for e in errs]) if errs else float("nan")
            med = statistics.median(errs) if errs else float("nan")
            mark = " <- default" if sb == args.pitched_setback_in else ""
            sites_txt = ", ".join(f"{k[:14]} {v}" for k, v in per.items())
            print(f"{sb:6g} in {hits:>5} of {len(errs):<3} {med_abs:>12.0%} {med:>+11.0%}{mark}  {sites_txt}")
    _snapshots(args, outcomes)
    print(f"\nFull report: {args.out}")
    return 0


def cmd_permits(args: argparse.Namespace) -> int:
    import random

    from .sources.permits import CITIES, pull_permits

    rows = []
    for city in args.city or list(CITIES):
        try:
            got = pull_permits(city, min_kw=args.min_kw, since_year=args.since, limit=args.limit)
        except Exception as exc:  # one portal down must not stop the others
            print(f"{city}: couldn't read permits ({type(exc).__name__}: {redact(exc)})", file=sys.stderr)
            continue
        if args.sample and len(got) > args.sample:
            got = random.Random(0).sample(got, args.sample)
        print(f"{city}: {len(got)} rooftop PV permits of {args.min_kw:g} kW or more")
        rows.extend(got)
    if not rows:
        print("No permits found.", file=sys.stderr)
        return 1
    _write_csv(rows, args.out, "name")
    print(f"Wrote {len(rows)} sites -> {args.out}")
    return 0


def _samples(results_path: str, truth_path: str, raw: bool) -> tuple[list[Sample], list[dict]]:
    truth = {r["building_id"]: float(r["true_kw"]) for r in _read_csv(truth_path) if r.get("true_kw")}
    rows = [r for r in _read_csv(results_path) if r["building_id"] in truth]
    col = "raw_kw" if raw else "dc_kw"
    return [Sample(r["segment"], float(r[col]), truth[r["building_id"]]) for r in rows], rows


def cmd_calibrate(args: argparse.Namespace) -> int:
    samples, _ = _samples(args.results, args.truth, raw=True)
    if len(samples) < 2:
        print("need at least 2 buildings with ground truth", file=sys.stderr)
        return 2
    report = {
        "uncalibrated": accuracy([s.predicted_kw for s in samples], [s.true_kw for s in samples]),
        "calibrated_cross_validated": cross_validate(samples, folds=args.folds, min_samples=args.min_samples),
    }
    cal = Calibrator(min_samples=args.min_samples).fit(samples)
    cal.save(args.out)
    report["factors"] = {k: {"factor": round(f, 4), "n": n} for k, (f, n) in cal.factors.items()}
    report["global_factor"] = round(cal.global_factor, 4)
    print(json.dumps(report, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    samples, rows = _samples(args.results, args.truth, raw=False)
    auto = [s for s, r in zip(samples, rows) if r["needs_review"] == "False"]
    report = {
        "all": accuracy([s.predicted_kw for s in samples], [s.true_kw for s in samples]),
        "auto_accepted": accuracy([s.predicted_kw for s in auto], [s.true_kw for s in auto]),
        "auto_accept_rate": len(auto) / len(samples) if samples else 0.0,
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rooftop-solar", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    design = argparse.ArgumentParser(add_help=False)
    design.add_argument("--calibration", help="calibration JSON from `calibrate`")
    design.add_argument("--google", action="store_true", help="cross-check with Google Solar API (billed per uncached call)")
    design.add_argument("--google-key-env", default="GOOGLE_SOLAR_API_KEY")
    design.add_argument("--google-cache", default=".cache/google_solar")
    design.add_argument("--google-max-points", type=int, default=9,
                        help="max Google lookups per building; large buildings are split by Google into pieces")
    design.add_argument("--no-equipment-detection", action="store_true",
                        help="don't look up rooftop equipment in Google's surface model (one billed dataLayers call per flat roof)")
    design.add_argument("--snapshots", default="",
                        help="save satellite PNGs with the layout drawn on for these sites (name parts, ';'-separated)")
    design.add_argument("--workers", type=int, default=8)
    design.add_argument("--include-north-faces", action="store_true",
                        help="keep Google panels on north-facing pitched roof faces (excluded by default)")
    design.add_argument("--min-panel-energy-ratio", type=float, default=0.6,
                        help="drop Google panels producing less than this fraction of the building's 90th-percentile panel")
    design.add_argument("--min-modules-per-structure", type=int, default=6,
                        help="structures that fit fewer modules than this in total are not designed")
    design.add_argument("--no-carports", action="store_true",
                        help="on parcels, skip carport rows and garages entirely")
    design.add_argument("--carport-min-energy-ratio", type=float, default=0.8,
                        help="count a carport only if its typical panel yields at least this fraction of the best roof panel")
    design.add_argument("--no-parcels", action="store_true", help="don't use Regrid parcel boundaries")
    design.add_argument("--regrid-token-file", default="~/.rooftop-solar/regrid_token",
                        help="file holding a Regrid API token (or set REGRID_TOKEN)")
    design.add_argument("--no-unit-points", action="store_true",
                        help="don't add buildings found under county per-unit address points")
    design.add_argument("--module-watts", type=float, default=550.0)
    design.add_argument("--module-length-m", type=float, default=2.278)
    design.add_argument("--module-width-m", type=float, default=1.134)
    design.add_argument("--flat-racking", choices=[r.value for r in Racking], default=Racking.EAST_WEST.value)
    design.add_argument("--flat-tilt-deg", type=float, default=10.0)
    design.add_argument("--gcr", type=float, default=0.90, help="ground coverage ratio for flat-roof racking")
    design.add_argument("--south-gcr-fixed", action="store_true", help="use --gcr for south racking instead of shading-derived spacing")
    design.add_argument("--section-gap-ft", type=float, default=4.0, help="IFC 1205.3.3 array separation (4 or 8 ft)")
    design.add_argument("--edge-setback-ft", type=float, default=0.0, help="wind/structural edge setback if larger than fire code")
    design.add_argument("--pitched-setback-in", type=float, default=36.0,
                        help="setback from pitched roof-plane edges in inches (36; some AHJs allow 18)")
    design.add_argument("--carport-min-kw", type=float, default=15.0,
                        help="count a detached garage or carport only if at least this many kW fit")
    design.add_argument("--no-residential-alternative", action="store_true", help="apply commercial rules to pitched R-2 roofs")

    s = sub.add_parser("size", parents=[design], help="size every building in a KML/KMZ or GeoJSON file")
    s.add_argument("--buildings", required=True, help="Google Earth KML/KMZ or GeoJSON of roofs, planes and obstructions")
    s.add_argument("--out", required=True, help="output CSV")
    s.add_argument("--layouts", help="optional placed-module output for QA (.kml for Google Earth, else GeoJSON)")
    s.set_defaults(func=cmd_size)

    ss = sub.add_parser("size-sites", parents=[design], help="size a CSV of addresses or coordinates")
    ss.add_argument("--sites", required=True, help="CSV with an address column, or latitude/longitude")
    ss.add_argument("--out", required=True, help="per-site output CSV")
    ss.add_argument("--buildings-out", help="per-building output CSV (default: <out>_buildings.csv)")
    ss.add_argument("--layouts", help="optional placed-module output for QA (.kml for Google Earth)")
    ss.add_argument("--geocoder", choices=["auto", "overture", "census", "google"], default="auto",
                    help="auto: Overture address points, then Census (free, US); google: billed")
    ss.add_argument("--search-m", type=float, default=40.0, help="max distance from the address point to a building")
    ss.add_argument("--campus-radius-m", type=float, default=0.0,
                    help="also size every building within this radius (multi-building properties); 0 = one building")
    ss.set_defaults(func=cmd_size_sites)

    a = sub.add_parser("accuracy", parents=[design], help="compare against known max-fit designs")
    a.add_argument("--truth", required=True, help="truth JSON or CSV (name, address, latitude, longitude, true_kw, module_w)")
    a.add_argument("--out", required=True, help="report CSV")
    a.add_argument("--layouts", help="optional KML of roofs and panels for checking matches in Google Earth")
    a.add_argument("--geocoder", choices=["auto", "google"], default="auto")
    a.add_argument("--search-m", type=float, default=40.0)
    a.add_argument("--campus-radius-m", type=float, default=0.0)
    a.add_argument("--pitched-setbacks", default="0,18,36",
                   help="also size at these pitched-roof setbacks (inches) and compare; '' to skip")
    a.add_argument("--energy-ratios", default="0,0.6,0.7,0.8,0.9",
                   help="comma-separated panel-yield cutoffs to compare (uses saved Google answers, no extra cost)")
    a.set_defaults(func=cmd_accuracy)

    pm = sub.add_parser("permits", help="pull installed rooftop PV systems from city permit open data (accuracy-test minimums)")
    pm.add_argument("--city", action="append", help="sf, la, austin, seattle, chicago, nyc (repeatable; default all)")
    pm.add_argument("--min-kw", type=float, default=30.0, help="skip smaller (residential-scale) systems")
    pm.add_argument("--since", type=int, default=2015, help="skip permits issued before this year")
    pm.add_argument("--limit", type=int, default=5000, help="max permit records read per dataset")
    pm.add_argument("--sample", type=int, default=40,
                    help="keep at most this many sites per city (each site costs Google/Regrid lookups); 0 = all")
    pm.add_argument("--out", required=True)
    pm.set_defaults(func=cmd_permits)

    c = sub.add_parser("calibrate", help="fit correction factors against real designs")
    c.add_argument("--results", required=True, help="CSV from `size`")
    c.add_argument("--truth", required=True, help="CSV with building_id,true_kw")
    c.add_argument("--out", required=True, help="calibration JSON")
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--min-samples", type=int, default=5)
    c.set_defaults(func=cmd_calibrate)

    e = sub.add_parser("evaluate", help="accuracy of a sized portfolio against ground truth")
    e.add_argument("--results", required=True)
    e.add_argument("--truth", required=True)
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
