"""Accuracy test: size sites with known max-fit designs and compare.

Truth file: JSON {site name: {"address": ..., "lat": .., "lon": .., "truths":
[{"kw": .., "module_w": .., "source": ..}, ...]}} or a CSV with columns
name, address, latitude, longitude, true_kw, module_w (one row per design).
A site can have several designs (different designers' max fits); a prediction
counts as within tolerance if it is within tolerance of any of them.

A design with "racking": "raised" (CSV column racking=raised) is compared with
the raised-racking MaxFit only; the others with the standard MaxFit.

A site with "kind": "floor" (CSV column kind=floor) has only designs sized to
load, budget or a goal, not to the roof. Those are minimums: absolute MaxFit
should come in at or above them, so they only feed the below-design check.

Predictions are made with one module size, so each design's kW is compared
after scaling the prediction by (design module W / tool module W). That assumes
similar module dimensions, which holds only roughly across 530-635 W modules.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .sites import Site, SiteOutcome, _occupancy_from_text


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
            entry = truth.setdefault(name, {"address": r.get("address", ""), "lat": None, "lon": None, "truths": [],
                                            "kind": r.get("kind") or "maxfit"})
            if r.get("latitude") and r.get("longitude"):
                entry["lat"], entry["lon"] = float(r["latitude"]), float(r["longitude"])
            entry["truths"].append({
                "kw": float(r["true_kw"]),
                "module_w": float(r["module_w"]) if r.get("module_w") else None,
                "source": r.get("source", ""),
                "racking": r.get("racking") or "standard",
            })
    return truth


def truth_sites(truth: dict[str, dict]) -> list[Site]:
    return [Site(name, t.get("address") or "", t.get("lat"), t.get("lon"), _occupancy_from_text(t.get("occupancy") or ""),
                 "", t.get("units")) for name, t in truth.items()]


def _best_error(pred_kw: float, truths: list[dict], module_w: float, floor: bool = False) -> tuple[float, dict]:
    """Signed error against the closest design (the largest, for floors), after module-wattage scaling."""
    best = None
    for t in truths:
        scaled = pred_kw * ((t.get("module_w") or module_w) / module_w)
        err = (scaled - t["kw"]) / t["kw"]
        if best is None or (err < best[0] if floor else abs(err) < abs(best[0])):
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
        floor = t.get("kind") == "floor"
        by_roof: dict[str, float] = {}
        for e in est:
            if e.primary:
                by_roof[e.primary.roof_type] = by_roof.get(e.primary.roof_type, 0.0) + e.dc_kw
        roof = max(by_roof, key=by_roof.get) if by_roof else ""
        row = {
            "site": name,
            "kind": "floor" if floor else "maxfit",
            "roof": roof,
            "manual_review": ";".join(o.manual_review) if o else "",
            "designs_kw": " / ".join(f"{d['kw']:g}" + (f"@{d['module_w']:g}W" if d.get("module_w") else "")
                                     + ("R" if d.get("racking") == "raised" else "") for d in t["truths"]),
            "buildings_found": len(o.buildings) if o else 0,
            "location": f"{o.geocode.source}:{o.geocode.precision}" if o and o.geocode else "",
        }
        raised_designs = [d for d in t["truths"] if d.get("racking") == "raised"]
        standard_designs = [d for d in t["truths"] if d.get("racking") != "raised"] or t["truths"]
        for label, kw in (("tool", final), ("raised", raised), ("footprint_only", geo), ("google", goo), ("google_unclipped", raw)):
            if kw:
                designs = (raised_designs or t["truths"]) if label == "raised" else standard_designs
                err, _ = _best_error(kw, designs, module_w, floor)
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
    maxfit = [r for r in rows if r["kind"] == "maxfit"]
    for label in ("tool", "raised", "footprint_only", "google", "google_unclipped"):
        sized = [r for r in maxfit if r[f"{label}_kw"] != ""]
        hits = sum(1 for r in sized if r[f"{label}_within"])
        lines.append(f"{label:17} {hits} of {len(sized)} MaxFit designs within ±{tolerance:.0%} ({len(maxfit) - len(sized)} not sized)")
    # Reference designs are often trimmed to the most cost-efficient roofs or sized
    # to load, i.e. lower bounds on absolute MaxFit: the meaningful failure is
    # coming in below them.
    sized = [r for r in rows if r["tool_err"]]
    below = [r for r in sized if float(r["tool_err"].rstrip("%")) / 100 < -tolerance]
    caught = [r["site"] for r in below if r["manual_review"]]
    # The outline-only estimate ignores equipment, so it is an upper bound for the
    # matched buildings. A design that doesn't fit even there was built on other
    # buildings (wrong match, campus, typo in the source): not a sizing miss.
    pct = lambda v: float(v.rstrip("%")) / 100 if v else None
    mismatch = [r["site"] for r in below if not r["manual_review"]
                and pct(r["footprint_only_err"]) is not None and pct(r["footprint_only_err"]) < -tolerance]
    missed = [r["site"] for r in below if not r["manual_review"] and r["site"] not in mismatch]
    flagged = sum(1 for r in rows if r["manual_review"])
    lines.append(f"\nDesigns as lower bounds (all {len(sized)} sites incl. {len(sized) - len([r for r in sized if r['kind'] == 'maxfit'])} "
                 f"load-sized floors): {len(sized) - len(below)} at or above design (within -{tolerance:.0%})")
    lines.append(f"  below design, flagged for manual review: {', '.join(caught) or 'none'}")
    lines.append(f"  below design, larger than the matched buildings' outline can hold (building match problem): "
                 f"{', '.join(mismatch) or 'none'}")
    errs = {r["site"]: r["tool_err"] for r in below}
    lines.append(f"  below design, NOT flagged (the misses that matter): "
                 f"{', '.join(f'{m} ({errs[m]})' for m in missed) or 'none'}")
    lines.append(f"Flagged for manual review: {flagged} of {len(rows)} sites")
    # Installed (or load-sized) systems against the tool's MaxFit: how much of the
    # roof typically gets built. Unflagged sites only.
    shares = sorted(1 / (1 + float(r["tool_err"].rstrip("%")) / 100) for r in sized
                    if r["kind"] == "floor" and not r["manual_review"] and float(r["tool_err"].rstrip("%")) > -100)
    if len(shares) >= 3:
        q = lambda f: shares[min(len(shares) - 1, int(f * len(shares)))]
        lines.append(f"\nInstalled/load-sized systems as a share of MaxFit ({len(shares)} unflagged floor sites): "
                     f"median {q(0.5):.0%}, middle half {q(0.25):.0%}-{q(0.75):.0%}")
    return "\n".join(lines)
