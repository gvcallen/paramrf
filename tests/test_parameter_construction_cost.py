"""Warm construction costs for parameters with priors (#319)."""

import timeit

import pytest

import pmrf as prf
from pmrf.materials import BulkConductor, DjordjevicSarkarDielectric
from pmrf.models import CoaxialLine, SchelkunoffCoaxialFormulation
from pmrf.stats.distributions import RelativeTruncatedNormal, Uniform


def _coaxial_line():
    return CoaxialLine(
        length=1.0, d_in=3.124e-3, d_out=8.328e-3,
        dielectric=DjordjevicSarkarDielectric(
            ep_r=1.33, tand=1e-4, f_low=1.5e4, f_high=1e11, f_ref=1e8,
        ),
        conductor=BulkConductor(sigma=5.8e7),
        formulation=SchelkunoffCoaxialFormulation(),
    )


@pytest.mark.parametrize(
    ("factory", "target_ms", "calls"),
    [
        (lambda: prf.Random(RelativeTruncatedNormal(50.0, 0.025)), 1.0, 100),
        (lambda: prf.Random(Uniform(0.0, 50.0), value=0.0), 0.5, 100),
        (lambda: prf.Bounded(0.0, 10.0, value=1.0), 0.5, 100),
        (_coaxial_line, 5.0, 20),
    ],
    ids=["truncated-normal", "uniform", "bounded", "coaxial-line"],
)
def test_warm_parameter_construction_cost(factory, target_ms, calls):
    # Each sample averages many calls. Use the best of three to reduce unrelated
    # CI contention, then allow a 3x margin over the issue's nominal CPU targets.
    for _ in range(3):
        factory()
    elapsed = min(timeit.repeat(factory, number=calls, repeat=3))
    per_call_ms = elapsed / calls * 1000
    assert per_call_ms < 3 * target_ms, (
        f"warm construction took {per_call_ms:.2f} ms per call "
        f"(nominal target {target_ms:.1f} ms, CI limit {3 * target_ms:.1f} ms)"
    )
