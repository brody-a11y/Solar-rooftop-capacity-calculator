"""Per-building pipeline: run estimators, cross-check them, calibrate, route to review.

At scale the goal is not that every automated number is within 10%. It is that the
ones auto-accepted are, and the rest are sent to a person. The cross-check between
two independent methods is what makes that routing possible.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from .calibration import Calibrator, segment_key
from .models import Building, Obstruction, SizingResult
from .sizing import GeometricEstimator
from .sources.google_solar import GoogleFilteredEstimator, GoogleInsights, GoogleLookupError, GoogleSolarClient, fetch_building


@dataclass(frozen=True)
class ReviewPolicy:
    """Starting thresholds. Tune them on your validation set: widen or narrow the
    band until auto-accepted sites meet the accuracy target."""

    # google_filtered / geometric. Geometric ignores unmapped clutter, so it should
    # be an upper bound; a ratio above ~1 means the two methods disagree on the roof.
    agree_low: float = 0.70
    agree_high: float = 1.05
    min_kw: float = 1.0


@dataclass
class SiteEstimate:
    building_id: str
    dc_kw: float
    raw_kw: float
    method: str
    calibration_factor: float
    calibration_source: str
    agreement_ratio: float | None
    needs_review: bool
    reasons: list[str] = field(default_factory=list)
    primary: SizingResult | None = None
    geometric: SizingResult | None = None
    google: SizingResult | None = None
    equipment_status: str = ""  # "", "found_N", or "failed:<reason>"

    def row(self, occupancy: str) -> dict:
        p = self.primary
        return {
            "building_id": self.building_id,
            "dc_kw": round(self.dc_kw, 2),
            "raw_kw": round(self.raw_kw, 2),
            "module_count": p.module_count if p else 0,
            "method": self.method,
            "segment": segment_key(self.method, occupancy, p.roof_type if p else ""),
            "calibration_factor": round(self.calibration_factor, 4),
            "calibration_source": self.calibration_source,
            "geometric_kw": round(self.geometric.dc_kw, 2) if self.geometric else "",
            "google_kw": round(self.google.dc_kw, 2) if self.google else "",
            "agreement_ratio": round(self.agreement_ratio, 3) if self.agreement_ratio is not None else "",
            "gross_roof_area_m2": round(p.gross_roof_area_m2, 1) if p else "",
            "usable_area_m2": round(p.usable_area_m2, 1) if p else "",
            "roof_type": p.roof_type if p else "",
            "needs_review": self.needs_review,
            "reasons": ";".join(self.reasons),
            "flags": ";".join(p.flags) if p else "",
        }


def estimate_site(
    building: Building,
    geometric: GeometricEstimator,
    google: GoogleFilteredEstimator | None = None,
    insights: GoogleInsights | None = None,
    calibrator: Calibrator | None = None,
    policy: ReviewPolicy = ReviewPolicy(),
) -> SiteEstimate:
    reasons: list[str] = []
    geo = geometric.estimate(building)
    goo = google.estimate(building, insights) if google and insights else None

    ratio = goo.dc_kw / geo.dc_kw if goo and geo.dc_kw > 0 else None
    google_usable = goo is not None and goo.module_count > 0 and not any(f.startswith("imagery_quality_") for f in goo.flags)
    if google_usable:
        primary = goo
        if ratio is not None and not (policy.agree_low <= ratio <= policy.agree_high):
            reasons.append(f"methods_disagree_ratio_{ratio:.2f}")
    else:
        primary = geo
        if goo is not None:
            reasons.append("google_unusable:" + ",".join(goo.flags or ["zero_panels"]))
        if not building.obstructions_mapped:
            reasons.append("single_method_no_obstruction_data")

    factor, source = (calibrator.factor_for(segment_key(primary.method, building.occupancy.value, primary.roof_type)) if calibrator else (1.0, "none"))
    if source == "none":
        reasons.append("uncalibrated")
    if primary.dc_kw < policy.min_kw:
        reasons.append("no_usable_roof_area")

    return SiteEstimate(
        building_id=building.id,
        dc_kw=primary.dc_kw * factor,
        raw_kw=primary.dc_kw,
        method=primary.method,
        calibration_factor=factor,
        calibration_source=source,
        agreement_ratio=ratio,
        needs_review=bool(reasons),
        reasons=reasons,
        primary=primary,
        geometric=geo,
        google=goo,
    )


def _size_one(job) -> SiteEstimate:
    building, geometric, insights, error, calibrator, policy = job
    google = GoogleFilteredEstimator(geometric) if insights else None
    try:
        est = estimate_site(building, geometric, google, insights, calibrator, policy)
    except Exception as exc:  # one bad geometry must not stop a batch: fall back to the outline
        try:
            est = estimate_site(building, geometric, None, None, calibrator, policy)
            est.reasons.insert(0, f"google_sizing_error:{type(exc).__name__}_sized_from_outline")
        except Exception as exc2:
            est = SiteEstimate(building.id, 0.0, 0.0, "error", 1.0, "none", None, True,
                               [f"sizing_error:{type(exc2).__name__}"])
        est.needs_review = True
    if error:
        est.reasons.insert(0, f"google_error:{error}")
        est.needs_review = True
    return est


def estimate_many(
    buildings: list[Building],
    geometric: GeometricEstimator,
    google_client: GoogleSolarClient | None = None,
    calibrator: Calibrator | None = None,
    policy: ReviewPolicy = ReviewPolicy(),
    workers: int = 8,
    google_max_points: int = 9,
    equipment_client=None,
) -> list[SiteEstimate]:
    """Google lookups on threads (network-bound), sizing on processes (CPU-bound;
    threads contend on the GIL and run slower than one worker)."""
    insights: list[GoogleInsights | None] = [None] * len(buildings)
    errors: list[str | None] = [None] * len(buildings)
    if google_client:
        def fetch(i):
            try:
                insights[i] = fetch_building(google_client, buildings[i].footprint, google_max_points)
            except GoogleLookupError as exc:
                errors[i] = str(exc)
            except Exception as exc:  # recorded per building; one bad lookup must not stop a batch
                errors[i] = type(exc).__name__
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, range(len(buildings))))
    notes: list[list[str]] = [[] for _ in buildings]
    status: list[str] = [""] * len(buildings)
    if equipment_client:
        # Rooftop equipment from Google's surface model, only where Google put
        # panels on a flat roof (condenser fields are a flat-roof problem).
        flat_deg = geometric.design.flat_pitch_threshold_deg
        buildings = list(buildings)

        def equip(i):
            ins = insights[i]
            if ins is None or not any(s.pitch_deg < flat_deg for s in ins.segments) or not ins.panels:
                return
            try:
                found = equipment_client.equipment(buildings[i].footprint)
            except Exception as exc:  # sizing goes ahead without it, flagged
                code = getattr(getattr(exc, "response", None), "status_code", None)
                detail = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(exc))[:40].strip("_")
                status[i] = f"failed:{f'http_{code}' if code else type(exc).__name__ + (f'_{detail}' if detail else '')}"
                notes[i].append(f"equipment_lookup_{status[i]}")
                return
            status[i] = f"found_{len(found)}"
            buildings[i] = replace(buildings[i], obstructions=buildings[i].obstructions
                                   + [Obstruction(g, "equipment") for g in found])
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:  # dataLayers rate-limits bursts
            list(pool.map(equip, range(len(buildings))))
    jobs = [(b, geometric, insights[i], errors[i], calibrator, policy) for i, b in enumerate(buildings)]
    procs = min(workers, os.cpu_count() or 1, len(jobs))
    if procs <= 1:
        results = [_size_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=procs) as pool:
            results = list(pool.map(_size_one, jobs, chunksize=max(1, len(jobs) // (procs * 4))))
    for est, extra, st in zip(results, notes, status):
        est.reasons.extend(extra)
        est.equipment_status = st
    return results
