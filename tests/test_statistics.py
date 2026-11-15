"""Summary statistics and percentile-bootstrap intervals (ADR-0009 revision 2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from swarm.evaluation.statistics import (
    BootstrapSpec,
    bootstrap_mean_ci,
    paired_difference,
    summarize,
)

SPEC = BootstrapSpec(resamples=2000, confidence=0.95, seed=0)


def test_summary_counts_and_moments() -> None:
    s = summarize([1.0, 2.0, None, 4.0], n_failed=2, spec=SPEC, metric="m")
    assert (s.n_seeds, s.n_failed, s.n_completed, s.n_defined, s.n_undefined) == (6, 2, 4, 3, 1)
    assert s.mean == pytest.approx(7 / 3)
    assert s.std == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
    low, high, mean = s.ci_low, s.ci_high, s.mean
    assert low is not None
    assert high is not None
    assert mean is not None
    assert 1.0 <= low <= mean <= high <= 4.0


def test_booleans_are_zero_one() -> None:
    s = summarize([True, False, True, True], n_failed=0, spec=SPEC, metric="mission_success")
    assert s.mean == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("values", "mean", "std", "ci"),
    [
        ([], None, None, None),
        ([None, None], None, None, None),
        ([3.0], 3.0, None, None),
        ([3.0, None], 3.0, None, None),
    ],
)
def test_undefined_statistics(
    values: list[float | None], mean: float | None, std: float | None, ci: None
) -> None:
    s = summarize(values, n_failed=0, spec=SPEC, metric="m")
    assert s.mean == mean
    assert s.std == std
    assert s.ci_low is ci
    assert s.ci_high is ci


def test_identical_values_give_zero_width_interval() -> None:
    s = summarize([0.5] * 10, n_failed=0, spec=SPEC, metric="m")
    assert (s.ci_low, s.ci_high, s.std) == (0.5, 0.5, 0.0)


def test_bootstrap_matches_an_independent_reference_implementation() -> None:
    values = np.array([0.1, 0.4, 0.4, 0.9, 1.0, 0.0, 0.3], dtype=np.float64)
    rng = SPEC.generator("recall")
    idx = rng.integers(0, values.size, size=(SPEC.resamples, values.size))
    means = values[idx].mean(axis=1)
    expected = np.quantile(means, [0.025, 0.975], method="linear")
    assert bootstrap_mean_ci(values, SPEC, "recall") == (float(expected[0]), float(expected[1]))


def test_bootstrap_is_deterministic_and_metric_keyed() -> None:
    # Continuous values: coarse discrete data can give identical quantiles from different draws.
    values = np.random.default_rng(0).random(25)
    assert SPEC.generator("a").random() != SPEC.generator("b").random()
    assert bootstrap_mean_ci(values, SPEC, "a") == bootstrap_mean_ci(values, SPEC, "a")
    assert bootstrap_mean_ci(values, SPEC, "a") != bootstrap_mean_ci(values, SPEC, "b")
    other_seed = BootstrapSpec(resamples=2000, seed=1)
    assert bootstrap_mean_ci(values, SPEC, "a") != bootstrap_mean_ci(values, other_seed, "a")


def test_generator_independent_of_unrelated_metrics() -> None:
    """Adding a metric never changes another metric's interval (keyed by name, not position)."""
    a = SPEC.generator("recall").random(4)
    b = SPEC.generator("recall").random(4)
    assert np.array_equal(a, b)


def test_bootstrap_coverage_is_near_nominal() -> None:
    """Sanity check of the method: ~95% of intervals cover the true mean of N(0.5, 0.2)."""
    rng = np.random.default_rng(123)
    spec = BootstrapSpec(resamples=1000, seed=7)
    hits = 0
    trials = 300
    for k in range(trials):
        sample = rng.normal(0.5, 0.2, size=40)
        ci = bootstrap_mean_ci(sample, spec, f"t{k}")
        assert ci is not None
        hits += ci[0] <= 0.5 <= ci[1]
    # Percentile intervals undercover slightly at n=40; accept 0.89-0.99 (fixed seeds, not flaky).
    assert 0.89 * trials <= hits <= 0.99 * trials


@given(values=st.lists(st.floats(-100, 100), min_size=2, max_size=40))
def test_interval_brackets_lie_within_data_range(values: list[float]) -> None:
    ci = bootstrap_mean_ci(np.array(values), SPEC, "m")
    assert ci is not None
    eps = 1e-9 * (1 + max(abs(v) for v in values))
    assert min(values) - eps <= ci[0] <= ci[1] <= max(values) + eps


def test_paired_difference_uses_only_jointly_defined_seeds() -> None:
    base = [0.2, 0.5, None, 0.4, 0.1]
    cand = [0.3, None, 0.9, 0.6, 0.1]
    p = paired_difference(base, cand, spec=SPEC, metric="recall")
    assert (p.n_pairs, p.n_excluded) == (3, 2)
    assert p.mean_difference == pytest.approx((0.1 + 0.2 + 0.0) / 3)


def test_paired_difference_requires_alignment() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_difference([1.0], [1.0, 2.0], spec=SPEC, metric="m")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resamples": 10}, "resamples: must be >= 100"),
        ({"resamples": 1000.0}, "resamples: must be an integer"),
        ({"confidence": 1.0}, "confidence: must be in"),
        ({"confidence": 0.0}, "confidence: must be in"),
        ({"seed": -1}, "seed: must be a non-negative"),
    ],
)
def test_invalid_bootstrap_spec(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BootstrapSpec(**kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------- M3.5 hypothesis tests


def test_sign_flip_exact_matches_hand_enumeration() -> None:
    from swarm.evaluation.statistics import sign_flip_p_value

    # 3 positive differences: only the all-+ and all-- patterns reach |mean| = 1, so p = 2/8.
    assert sign_flip_p_value([1.0, 1.0, 1.0]) == 0.25
    assert sign_flip_p_value([0.0, 0.0]) == 1.0
    assert sign_flip_p_value([]) is None
    # zeros count in the mean's denominator but do not change the p-value pattern count
    assert sign_flip_p_value([2.0, 0.0, 0.0]) == 1.0


def test_sign_flip_monte_carlo_is_seeded_and_never_zero() -> None:
    from swarm.evaluation.statistics import sign_flip_p_value

    diffs = [1.0] * 30
    p = sign_flip_p_value(diffs, resamples=1000, seed=3)
    assert p == sign_flip_p_value(diffs, resamples=1000, seed=3)
    assert p is not None
    assert 0.0 < p <= 2 / 1001
    noisy = [(-1.0) ** k for k in range(40)]  # mean 0
    q = sign_flip_p_value(noisy, resamples=1000, seed=0)
    assert q is not None
    assert q > 0.5


def test_holm_adjustment_is_monotone_and_skips_none() -> None:
    from swarm.evaluation.statistics import holm_adjust

    adjusted = holm_adjust([0.01, None, 0.04, 0.03])
    assert adjusted[1] is None
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[3] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)  # monotone: max(0.04 * 1, 0.06)
    assert holm_adjust([0.9, 0.8]) == [pytest.approx(1.0), pytest.approx(1.0)]


def test_cohens_dz() -> None:
    from swarm.evaluation.statistics import cohens_dz

    assert cohens_dz([1.0, 3.0]) == pytest.approx(2.0 / 2**0.5)
    assert cohens_dz([1.0, 1.0]) is None
    assert cohens_dz([1.0]) is None


def test_bh_adjustment() -> None:
    from swarm.evaluation.statistics import bh_adjust

    adjusted = bh_adjust([0.01, 0.04, None, 0.03])
    assert adjusted[2] is None
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[3] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.04)
