"""Command line: size a portfolio, calibrate against real designs, evaluate accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from .calibration import Calibrator, Sample, accuracy, cross_validate
from .fire_code import FireCodeRules
from .models import Module, Racking
from .pipeline import ReviewPolicy, estimate_site
from .sizing import DesignConfig, GeometricEstimator
from .sources.geojson_io import load_buildings, write_layouts
from .sources.google_solar import GoogleFilteredEstimator, GoogleInsights, GoogleSolarClient


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cmd_size(args: argparse.Namespace) -> int:
    rules = FireCodeRules(
        section_gap_ft=args.section_gap_ft,
        residential_alternative_for_pitched_r2=not args.no_residential_alternative,
    )
    design = DesignConfig(
        module=Module(args.module_length_m, args.module_width_m, args.module_watts),
        flat_racking=Racking(args.flat_racking),
        flat_tilt_deg=args.flat_tilt_deg,
        east_west_gcr=args.gcr,
        south_gcr=args.gcr if args.flat_racking == Racking.SOUTH.value and args.south_gcr_fixed else None,
        edge_setback_ft=args.edge_setback_ft,
    )
    geometric = GeometricEstimator(rules, design)
    calibrator = Calibrator.load(args.calibration) if args.calibration else None
    policy = ReviewPolicy()

    client = google = None
    if args.google:
        key = os.environ.get(args.google_key_env)
        if not key:
            print(f"--google set but ${args.google_key_env} is empty", file=sys.stderr)
            return 2
        client = GoogleSolarClient(key, cache_dir=args.google_cache)
        google = GoogleFilteredEstimator(geometric)

    buildings = load_buildings(args.buildings)

    def run(b):
        insights = None
        if client:
            pt = b.footprint.representative_point()
            try:
                insights = GoogleInsights.from_response(client.building_insights(pt.y, pt.x))
            except Exception as exc:  # recorded per site; one bad lookup must not stop a batch
                est = estimate_site(b, geometric, None, None, calibrator, policy)
                est.reasons.insert(0, f"google_error:{type(exc).__name__}")
                est.needs_review = True
                return b, est
        return b, estimate_site(b, geometric, google, insights, calibrator, policy)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, buildings))

    rows = [est.row(b.occupancy.value) for b, est in results]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["building_id"])
        w.writeheader()
        w.writerows(rows)
    if args.layouts:
        write_layouts([est.primary for _, est in results if est.primary], args.layouts)
    n_review = sum(r["needs_review"] for r in rows)
    print(f"sized {len(rows)} buildings -> {args.out}; {n_review} flagged for review")
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

    s = sub.add_parser("size", help="size every building in a GeoJSON file")
    s.add_argument("--buildings", required=True, help="GeoJSON of footprints / roof planes / obstructions")
    s.add_argument("--out", required=True, help="output CSV")
    s.add_argument("--layouts", help="optional GeoJSON of placed modules for QA")
    s.add_argument("--calibration", help="calibration JSON from `calibrate`")
    s.add_argument("--google", action="store_true", help="cross-check with Google Solar API (billed per uncached call)")
    s.add_argument("--google-key-env", default="GOOGLE_SOLAR_API_KEY")
    s.add_argument("--google-cache", default=".cache/google_solar")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--module-watts", type=float, default=550.0)
    s.add_argument("--module-length-m", type=float, default=2.278)
    s.add_argument("--module-width-m", type=float, default=1.134)
    s.add_argument("--flat-racking", choices=[r.value for r in Racking], default=Racking.EAST_WEST.value)
    s.add_argument("--flat-tilt-deg", type=float, default=10.0)
    s.add_argument("--gcr", type=float, default=0.90, help="ground coverage ratio for flat-roof racking")
    s.add_argument("--south-gcr-fixed", action="store_true", help="use --gcr for south racking instead of shading-derived spacing")
    s.add_argument("--section-gap-ft", type=float, default=4.0, help="IFC 1205.3.3 array separation (4 or 8 ft)")
    s.add_argument("--edge-setback-ft", type=float, default=0.0, help="wind/structural edge setback if larger than fire code")
    s.add_argument("--no-residential-alternative", action="store_true", help="apply commercial rules to pitched R-2 roofs")
    s.set_defaults(func=cmd_size)

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
