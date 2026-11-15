"""Multi-seed summary statistics and percentile-bootstrap confidence intervals (ADR-0009).

Methodology (recorded verbatim in every experiment result by :meth:`BootstrapSpec.to_dict`):

* statistic: arithmetic mean of the defined (non-null) per-episode values; booleans as 0/1;
* spread: sample standard deviation, ``ddof = 1``;
* interval: percentile bootstrap of the mean. Resample the ``n`` defined values with replacement
  ``resamples`` times, take each resample's mean, and report the ``(1 - c) / 2`` and
  ``(1 + c) / 2`` quantiles (NumPy ``linear`` method);
* generator: ``default_rng(SeedSequence(seed, spawn_key=(crc32(metric_name),)))``, independent
  of every episode seed and stable when metrics are added or removed;
* undefined: standard deviation and interval are ``None`` when ``n < 2``; the mean is ``None``
  when ``n = 0``.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

_BLOCK: Final[int] = 1000  # resamples generated per block, bounding memory for large n

Value = float | int | bool | None


@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    """Bootstrap settings. Defaults: 10 000 resamples, 95% interval, seed 0."""

    resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int):
            raise ValueError(f"resamples: must be an integer, got {self.resamples!r}")
        if self.resamples < 100:
            raise ValueError(f"resamples: must be >= 100, got {self.resamples}")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, int | float):
            raise ValueError(f"confidence: must be a number, got {self.confidence!r}")
        if not (0.0 < float(self.confidence) < 1.0) or not math.isfinite(self.confidence):
            raise ValueError(f"confidence: must be in (0, 1), got {self.confidence!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed: must be a non-negative integer, got {self.seed!r}")

    def generator(self, metric: str) -> np.random.Generator:
        """The bootstrap generator for one metric (see module docstring)."""
        key = zlib.crc32(metric.encode("utf-8"))
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(key,)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "percentile bootstrap of the mean",
            "resamples": self.resamples,
            "confidence": self.confidence,
            "seed": self.seed,
            "quantile_method": "numpy linear",
            "generator": "default_rng(SeedSequence(seed, spawn_key=(crc32(metric_name),)))",
            "std": "sample standard deviation, ddof=1",
            "booleans": "counted as 0/1",
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate of one metric over an experiment's episodes.

    ``n_seeds = n_completed + n_failed`` and ``n_completed = n_defined + n_undefined``.
    """

    n_seeds: int
    n_failed: int
    n_completed: int
    n_defined: int
    n_undefined: int
    mean: float | None
    std: float | None
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedSummary:
    """Mean of per-seed differences (candidate - baseline) with a bootstrap interval."""

    n_pairs: int
    n_excluded: int
    mean_difference: float | None
    std: float | None
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_array(values: Sequence[Value]) -> npt.NDArray[np.float64]:
    return np.array([float(v) for v in values if v is not None], dtype=np.float64)


def bootstrap_mean_ci(
    values: npt.NDArray[np.float64], spec: BootstrapSpec, metric: str
) -> tuple[float, float] | None:
    """Percentile-bootstrap interval of the mean, or ``None`` if fewer than two values."""
    n = int(values.size)
    if n < 2:
        return None
    rng = spec.generator(metric)
    means = np.empty(spec.resamples, dtype=np.float64)
    for start in range(0, spec.resamples, _BLOCK):
        rows = min(_BLOCK, spec.resamples - start)
        idx = rng.integers(0, n, size=(rows, n))
        means[start : start + rows] = values[idx].mean(axis=1)
    alpha = (1.0 - float(spec.confidence)) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def summarize(
    values: Sequence[Value], *, n_failed: int, spec: BootstrapSpec, metric: str
) -> MetricSummary:
    """Summarize per-episode values of one metric from the *completed* episodes.

    Args:
        values: one entry per completed episode; ``None`` marks an undefined value.
        n_failed: number of failed episodes (no value at all), reported, never hidden.
    """
    data = _as_array(values)
    n = int(data.size)
    ci = bootstrap_mean_ci(data, spec, metric)
    return MetricSummary(
        n_seeds=len(values) + n_failed,
        n_failed=n_failed,
        n_completed=len(values),
        n_defined=n,
        n_undefined=len(values) - n,
        mean=float(data.mean()) if n else None,
        std=float(data.std(ddof=1)) if n >= 2 else None,
        ci_low=None if ci is None else ci[0],
        ci_high=None if ci is None else ci[1],
    )


def paired_difference(
    baseline: Sequence[Value], candidate: Sequence[Value], *, spec: BootstrapSpec, metric: str
) -> PairedSummary:
    """Paired per-seed differences ``candidate - baseline``.

    Both sequences are aligned by seed. A pair is used only if both values are defined;
    other pairs are counted in ``n_excluded``.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired_difference: sequences must be aligned by seed (equal length)")
    diffs = [
        float(c) - float(b)
        for b, c in zip(baseline, candidate, strict=True)
        if b is not None and c is not None
    ]
    data = np.array(diffs, dtype=np.float64)
    n = int(data.size)
    ci = bootstrap_mean_ci(data, spec, metric)
    return PairedSummary(
        n_pairs=n,
        n_excluded=len(baseline) - n,
        mean_difference=float(data.mean()) if n else None,
        std=float(data.std(ddof=1)) if n >= 2 else None,
        ci_low=None if ci is None else ci[0],
        ci_high=None if ci is None else ci[1],
    )


# ----------------------------------------------------------------- hypothesis tests (M3.5)

EXACT_SIGN_FLIP_MAX_N: Final[int] = 16


def sign_flip_p_value(
    differences: Sequence[float], *, resamples: int = 20_000, seed: int = 0
) -> float | None:
    """Two-sided paired randomization (sign-flip) test of a zero mean difference.

    Under the null each per-seed difference is equally likely to have either sign. The statistic
    is ``|mean|``. With at most :data:`EXACT_SIGN_FLIP_MAX_N` non-zero differences all sign
    patterns are enumerated (exact); otherwise ``resamples`` patterns are drawn from
    ``default_rng(seed)`` and ``p = (1 + #{stat >= observed}) / (1 + resamples)``, which is never
    0. Returns ``None`` with no differences, and 1.0 when every difference is 0.
    """
    data = np.asarray(differences, dtype=np.float64)
    if data.size == 0:
        return None
    observed = abs(float(data.mean()))
    nonzero = data[data != 0.0]
    if nonzero.size == 0:
        return 1.0
    tol = 1e-12 * max(1.0, observed)
    n = data.size
    if nonzero.size <= EXACT_SIGN_FLIP_MAX_N:
        patterns = ((np.arange(2**nonzero.size)[:, None] >> np.arange(nonzero.size)) & 1) * 2 - 1
        stats = np.abs((patterns * nonzero).sum(axis=1) / n)
        return float(np.mean(stats >= observed - tol))
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, resamples, _BLOCK):
        rows = min(_BLOCK, resamples - start)
        signs = rng.integers(0, 2, size=(rows, nonzero.size)) * 2 - 1
        hits += int(np.sum(np.abs((signs * nonzero).sum(axis=1) / n) >= observed - tol))
    return (1 + hits) / (1 + resamples)


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (family-wise error control), in input order.

    ``None`` entries (untestable) are passed through and do not count towards the family size.
    """
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def cohens_dz(differences: Sequence[float]) -> float | None:
    """Standardized paired effect size: mean difference / sample SD of the differences.

    ``None`` with fewer than two differences or zero spread.
    """
    data = np.asarray(differences, dtype=np.float64)
    if data.size < 2:
        return None
    sd = float(data.std(ddof=1))
    return float(data.mean()) / sd if sd > 0.0 else None


def bh_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini-Hochberg adjusted p-values (false-discovery-rate control), in input order.

    ``None`` entries are passed through and do not count towards the family size.
    """
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        p, i = indexed[rank]
        running = min(running, p * m / (rank + 1))
        adjusted[i] = min(1.0, running)
    return adjusted
