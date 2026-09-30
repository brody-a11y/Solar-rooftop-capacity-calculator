"""Calibrate estimates against ground-truth designs and measure accuracy honestly.

Ground truth should be the max-fill DC size from real engineered layouts (permit
sets or design-tool max-fill runs), not installed sizes limited by budget,
interconnection or load.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


def segment_key(method: str, occupancy: str, roof_type: str) -> str:
    return f"{method}|{occupancy}|{roof_type}"


@dataclass
class Sample:
    key: str
    predicted_kw: float
    true_kw: float


def accuracy(pred: list[float], truth: list[float], tolerance: float = 0.10) -> dict:
    errs = [(p - t) / t for p, t in zip(pred, truth) if t > 0]
    if not errs:
        return {"n": 0}
    abs_errs = sorted(abs(e) for e in errs)
    p90_idx = min(len(abs_errs) - 1, int(round(0.9 * (len(abs_errs) - 1))))
    return {
        "n": len(errs),
        f"within_{int(tolerance * 100)}pct": sum(a <= tolerance for a in abs_errs) / len(errs),
        "median_abs_pct_err": statistics.median(abs_errs),
        "mean_abs_pct_err": statistics.fmean(abs_errs),
        "p90_abs_pct_err": abs_errs[p90_idx],
        "median_signed_pct_err": statistics.median(errs),
    }


@dataclass
class Calibrator:
    """Multiplicative correction per segment: factor = median(true / predicted).

    Segments with fewer than `min_samples` fall back to the global factor.
    """

    min_samples: int = 5
    factors: dict[str, tuple[float, int]] = field(default_factory=dict)
    global_factor: float = 1.0
    global_n: int = 0

    def fit(self, samples: list[Sample]) -> "Calibrator":
        usable = [s for s in samples if s.predicted_kw > 0 and s.true_kw > 0]
        if not usable:
            raise ValueError("no samples with positive predicted and true kW")
        self.global_factor = statistics.median(s.true_kw / s.predicted_kw for s in usable)
        self.global_n = len(usable)
        by_key: dict[str, list[float]] = {}
        for s in usable:
            by_key.setdefault(s.key, []).append(s.true_kw / s.predicted_kw)
        self.factors = {k: (statistics.median(v), len(v)) for k, v in by_key.items() if len(v) >= self.min_samples}
        return self

    def factor_for(self, key: str) -> tuple[float, str]:
        if key in self.factors:
            return self.factors[key][0], "segment"
        if self.global_n:
            return self.global_factor, "global"
        return 1.0, "none"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "min_samples": self.min_samples,
                    "global_factor": self.global_factor,
                    "global_n": self.global_n,
                    "factors": {k: {"factor": f, "n": n} for k, (f, n) in self.factors.items()},
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        d = json.loads(Path(path).read_text())
        return cls(
            min_samples=d["min_samples"],
            factors={k: (v["factor"], v["n"]) for k, v in d["factors"].items()},
            global_factor=d["global_factor"],
            global_n=d["global_n"],
        )


def cross_validate(samples: list[Sample], folds: int = 5, min_samples: int = 5, tolerance: float = 0.10) -> dict:
    """Out-of-sample accuracy of calibrated predictions (k-fold, deterministic split).

    This is the number to quote. In-sample accuracy after calibration overstates it.
    """
    samples = [s for s in samples if s.predicted_kw > 0 and s.true_kw > 0]
    folds = max(2, min(folds, len(samples)))
    pred, truth = [], []
    for f in range(folds):
        train = [s for i, s in enumerate(samples) if i % folds != f]
        test = [s for i, s in enumerate(samples) if i % folds == f]
        if not train or not test:
            continue
        cal = Calibrator(min_samples=min_samples).fit(train)
        for s in test:
            factor, _ = cal.factor_for(s.key)
            pred.append(s.predicted_kw * factor)
            truth.append(s.true_kw)
    return accuracy(pred, truth, tolerance)
