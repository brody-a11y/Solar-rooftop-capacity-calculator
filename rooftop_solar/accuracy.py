"""Accuracy test: size sites with known max-fit designs and compare.

Truth file: JSON {site name: {"address": ..., "lat": .., "lon": .., "truths":
[{"kw": .., "module_w": .., "source": ..}, ...]}} or a CSV with columns
name, address, latitude, longitude, true_kw, module_w (one row per design).
A site can have several designs (different designers' max fits); a prediction
counts as within tolerance if it is within tolerance of any of them.

Predictions are made with one module size, so each design's kW is compared
after scaling the prediction by (design module W / tool module W). That assumes
similar module dimensions, which holds only roughly across 530-635 W modules.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .sites import Site, SiteOutcome


def load_truth(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    truth: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            r = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            name = r.get("name") or r.get("site") or r.get("property")
            if not name or not r.get("true_kw"):
                continue
            entry = truth.setdefault(name, {"address": r.get("address", ""), "lat": None, "lon": None, "truths": []})
            if r.get("latitude") and r.get("longitude"):
                entry["lat"], entry["lon"] = float(r["latitude"]), float(r["longitude"])
            entry["truths"].append({
                "kw": float(r["true_kw"]),
                "module_w": float(r["module_w"]) if r.get("module_w") else None,
                "source": r.get("source", ""),
            })
    return truth


def truth_sites(truth: dict[str, dict]) -> list[Site]:
    return [Site(name, t.get("address") or "", t.get("lat"), t.get("lon")) for name, t in truth.items()]


def _best_error(pred_kw: float, truths: list[dict], module_w: float) -> tuple[float, dict]:
    """Signed error against the closest design, after module-wattage scaling."""
    best = None
    for t in truths:
        scaled = pred_kw * ((t.get("module_w") or module_w) / module_w)
        err = (scaled - t["kw"]) / t["kw"]
        if best is None or abs(err) < abs(best[0]):
            best = (err, t)
    return best


def compare(truth: dict[str, dict], outcomes: list[SiteOutcome], module_w: float, tolerance: float = 0.10) -> list[dict]:
    rows = []
    by_id = {o.site.id: o for o in outcomes}
    for name, t in truth.items():
        o = by_id.get(name)
        est = o.rooftop_estimates() if o else []  # designs in the test set are rooftop-only
        final = sum(e.dc_kw for e in est)
        geo = sum(e.geometric.dc_kw for e in est if e.geometric)
        goo = sum(e.google.dc_kw for e in est if e.google) if any(e.google for e in est) else None
        raw = sum(e.google.details.get("google_unclipped_kw", 0) for e in est if e.google) if goo is not None else None
        raised = (final + o.raised_extra_kw(rooftop_only=True)) if o and final else None
        row = {
            "site": name,
            "designs_kw": " / ".join(f"{d['kw']:g}" + (f"@{d['module_w']:g}W" if d.get("module_w") else "") for d in t["truths"]),
            "buildings_found": len(o.buildings) if o else 0,
            "location": f"{o.geocode.source}:{o.geocode.precision}" if o and o.geocode else "",
        }
        for label, kw in (("tool", final), ("raised", raised), ("footprint_only", geo), ("google", goo), ("google_unclipped", raw)):
            if kw:
                err, _ = _best_error(kw, t["truths"], module_w)
                row[f"{label}_kw"] = round(kw, 1)
                row[f"{label}_err"] = f"{err:+.0%}"
                row[f"{label}_within"] = abs(err) <= tolerance
            else:
                row[f"{label}_kw"], row[f"{label}_err"], row[f"{label}_within"] = "", "", ""
        row["google_imagery_date"] = ";".join(sorted({e.google.details.get("imagery_date", "") for e in est if e.google}))
        row["reasons"] = o.row()["reasons"] if o else "not run"
        rows.append(row)
    return rows


def summary(rows: list[dict], tolerance: float = 0.10) -> str:
    lines = []
    for label in ("tool", "raised", "footprint_only", "google", "google_unclipped"):
        sized = [r for r in rows if r[f"{label}_kw"] != ""]
        hits = sum(1 for r in sized if r[f"{label}_within"])
        lines.append(f"{label:17} {hits} of {len(sized)} sized sites within ±{tolerance:.0%} ({len(rows) - len(sized)} not sized)")
    # Reference designs are often trimmed to the most cost-efficient roofs, i.e. lower
    # bounds on absolute MaxFit: the meaningful failure is coming in below them.
    sized = [r for r in rows if r["tool_err"]]
    below = [r["site"] for r in sized if float(r["tool_err"].rstrip("%")) / 100 < -tolerance]
    lines.append(f"\nIf designs are lower bounds (cost-trimmed): {len(sized) - len(below)} of {len(sized)} sites at or above "
                 f"design (within -{tolerance:.0%}); below: {', '.join(below) or 'none'}")
    return "\n".join(lines)
