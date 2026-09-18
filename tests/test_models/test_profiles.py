"""Profile shapes: the target-agnostic vocabulary profiled lines consume (ADR-0004)."""
import numpy as np
import pytest
import equinox as eqx
import jax
import jax.numpy as jnp

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import ConstantDielectric, BulkConductor
from pmrf.models import (
    AbstractProfile,
    LinearProfile,
    ExponentialProfile,
    KlopfensteinProfile,
    MicrostripLine,
)


T = jnp.linspace(0.0, 1.0, 11)


# ---------------------------------------------------------
# The abstract base
# ---------------------------------------------------------

def test_profile_is_a_module_but_not_a_model():
    """A profile has no ports and no S-parameters, so it is a Module, not a Model."""
    prof = LinearProfile(start=1.0, end=2.0)
    assert prf.is_module(prof)
    assert not prf.is_model(prof)
    assert isinstance(prof, AbstractProfile)


def test_abstract_profile_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractProfile()


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

def test_linear_endpoints():
    prof = LinearProfile(start=2e-3, end=8e-3)
    assert prof.evaluate(0.0) == pytest.approx(2e-3)
    assert prof.evaluate(1.0) == pytest.approx(8e-3)


def test_exponential_endpoints():
    prof = ExponentialProfile(start=2e-3, end=8e-3)
    assert prof.evaluate(0.0) == pytest.approx(2e-3)
    assert prof.evaluate(1.0) == pytest.approx(8e-3)


def test_linear_midpoint_is_the_arithmetic_mean():
    prof = LinearProfile(start=2.0, end=8.0)
    assert prof.evaluate(0.5) == pytest.approx(5.0)


def test_exponential_midpoint_is_the_geometric_mean():
    prof = ExponentialProfile(start=2.0, end=8.0)
    assert prof.evaluate(0.5) == pytest.approx(4.0)


def test_t_zero_is_port_one():
    """The documented convention: `t = 0` at port 1, `t = 1` at port 2. Reversing a
    shape swaps its coefficients; there is no orientation flag."""
    forward = ExponentialProfile(start=2e-3, end=8e-3)
    reversed_ = ExponentialProfile(start=8e-3, end=2e-3)
    assert forward.evaluate(T) == pytest.approx(reversed_.evaluate(T[::-1]))


# ---------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------

@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
@pytest.mark.parametrize('start, end', [(2e-3, 8e-3), (8e-3, 2e-3)])
def test_monotonic_between_endpoints(cls, start, end):
    """Every shipped shape is monotone in `t`, increasing or decreasing with the
    coefficients."""
    values = np.asarray(cls(start=start, end=end).evaluate(jnp.linspace(0.0, 1.0, 101)))
    diffs = np.diff(values)
    assert np.all(diffs > 0) if end > start else np.all(diffs < 0)


# ---------------------------------------------------------
# Elementwise evaluation
# ---------------------------------------------------------

@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_evaluate_is_elementwise(cls):
    """`evaluate` is elementwise, so a container can pass the whole midpoint array in
    one call and get the same answers as point-by-point evaluation."""
    prof = cls(start=2e-3, end=8e-3)
    ts = jnp.linspace(0.0, 1.0, 7)

    batched = prof.evaluate(ts)
    assert batched.shape == ts.shape

    one_by_one = jnp.stack([prof.evaluate(t) for t in ts])
    assert batched == pytest.approx(one_by_one, rel=1e-12)


@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_evaluate_preserves_shape(cls):
    """Elementwise means any shape in, the same shape out."""
    prof = cls(start=2e-3, end=8e-3)
    ts = jnp.linspace(0.0, 1.0, 6).reshape(2, 3)
    assert prof.evaluate(ts).shape == (2, 3)
    assert jnp.shape(prof.evaluate(0.5)) == ()


@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_evaluate_is_jittable_and_vmappable(cls):
    """Profiles are evaluated inside traced code, so they must survive both."""
    prof = cls(start=2e-3, end=8e-3)
    ts = jnp.linspace(0.0, 1.0, 5)

    jitted = jax.jit(lambda p, t: p.evaluate(t))(prof, ts)
    mapped = jax.vmap(prof.evaluate)(ts)
    assert jitted == pytest.approx(prof.evaluate(ts), rel=1e-12)
    assert mapped == pytest.approx(prof.evaluate(ts), rel=1e-12)


@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_evaluate_is_differentiable_in_its_coefficients(cls):
    """Coefficients are what the user fits, so the shape must carry a gradient."""
    def objective(start):
        return cls(start=start, end=8e-3).evaluate(0.5)

    grad = jax.grad(objective)(2e-3)
    assert jnp.isfinite(grad)
    assert grad != 0.0


# ---------------------------------------------------------
# Coefficients are ordinary parameters
# ---------------------------------------------------------

@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_coefficients_are_named_parameters(cls):
    """Coefficients are declared with `prf.param`, so they are listed by `prf.params`
    and take part in fitting like any other parameter."""
    names = prf.params(cls(start=2e-3, end=8e-3))
    assert 'start' in names
    assert 'end' in names


@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_plain_values_and_constrained_parameters_are_interchangeable(cls):
    """No profile-specific machinery: a plain float and a constrained parameter
    describe the same shape."""
    plain = cls(start=2e-3, end=8e-3)
    constrained = cls(
        start=prf.Bounded(1e-3, 5e-3, value=2e-3),
        end=prf.Bounded(5e-3, 9e-3, value=8e-3),
    )
    assert constrained.evaluate(T) == pytest.approx(plain.evaluate(T))

    # A plain value is a fixed parameter; a constrained one is free and keeps its own
    # bounds. Neither needed any profile-side handling.
    assert prf.params(plain, 'start')['start'].fixed
    assert not prf.params(constrained, 'start')['start'].fixed
    assert prf.params(constrained, 'start')['start'].bounds == (1e-3, 5e-3)


@pytest.mark.parametrize('cls', [ExponentialProfile, KlopfensteinProfile])
def test_coefficients_carry_their_constraints(cls):
    """A shape defined through a ratio of its coefficients is undefined for a
    non-positive one, and the field constraint says so at construction."""
    with pytest.raises(eqx.EquinoxRuntimeError, match='Positive'):
        cls(start=-2e-3, end=8e-3)


def test_klopfenstein_ripple_is_constrained_to_the_unit_interval():
    with pytest.raises(eqx.EquinoxRuntimeError, match='Interval'):
        KlopfensteinProfile(start=50.0, end=100.0, ripple=1.5)


# ---------------------------------------------------------
# A profile carries no target or family knowledge
# ---------------------------------------------------------

def _physical(model, name):
    """The physical value of one named parameter of `model`."""
    return float(prf.params(model, name)[name].physical_value)


def test_one_profile_instance_drives_targets_of_different_physical_kinds():
    """The central constraint of ADR-0004: `evaluate` returns a plain value in the
    units of whatever it is attached to. One instance drives a width in metres and a
    dimensionless permittivity, with no profile-side change."""
    shape = LinearProfile(start=1.0, end=4.0)
    ts = jnp.array([0.0, 0.5, 1.0])
    values = shape.evaluate(ts)

    freq = Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')
    base = MicrostripLine(
        w=1e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=10e-3,
    )

    # The same three numbers, read once as millimetres of width and once as a
    # relative permittivity. The profile is not told which, and is not changed.
    as_width = [prf.update(base, {'w': float(v) * 1e-3}) for v in values]
    as_ep_r = [prf.update(base, {'substrate.dielectric.ep_r': float(v)}) for v in values]

    widths = [_physical(m, 'w') for m in as_width]
    perms = [_physical(m, 'substrate.dielectric.ep_r') for m in as_ep_r]

    assert widths == pytest.approx([1e-3, 2.5e-3, 4e-3])
    assert perms == pytest.approx([1.0, 2.5, 4.0])

    # And both are physically usable: the impedance moves in opposite directions,
    # confirming the two readings are genuinely different physical kinds.
    z_width = [float(jnp.real(m.zc_and_gammaL(freq)[0][0])) for m in as_width]
    z_ep_r = [float(jnp.real(m.zc_and_gammaL(freq)[0][0])) for m in as_ep_r]
    assert z_width[0] > z_width[-1]
    assert z_ep_r[0] > z_ep_r[-1]


# ---------------------------------------------------------
# Klopfenstein against published values
# ---------------------------------------------------------

def _skrf_klopfenstein(start, end, ripple, ts):
    """The shape scikit-rf's `Klopfenstein` taper produces, sampled at `ts`.

    scikit-rf's `rmax` is the ripple relative to $\\Gamma_0$, the same convention as
    `KlopfensteinProfile.ripple`.
    """
    from skrf.taper import Klopfenstein

    taper = Klopfenstein(
        med=None, param='z0', start=start, stop=end, n_sections=2,
        length=1.0, length_unit='m', f_kw={'rmax': ripple},
    )
    # A unit-length taper, so position along it is already the normalised `t`. The
    # endpoints are excluded by the caller: scikit-rf's `heaviside(0) = 1` convention
    # makes its `f` jump by a factor of `e` at `t = 1` exactly.
    return np.asarray(taper.f(np.asarray(ts), 1.0, start, end, rmax=ripple))


def test_klopfenstein_matches_scikit_rf_shape():
    """Cross-check of the $\\phi$ quadrature against an independent implementation of
    the same published formula. Both sides are the design solution, so this checks the
    shape, not the simulator."""
    start, end, ripple = 50.0, 100.0, 0.05
    ts = np.linspace(0.02, 0.98, 25)

    ours = np.asarray(KlopfensteinProfile(start=start, end=end, ripple=ripple).evaluate(jnp.asarray(ts)))
    theirs = _skrf_klopfenstein(start, end, ripple, ts)

    # scikit-rf integrates each point with adaptive quadrature; we use fixed-order
    # Gauss-Legendre on an analytic integrand, so agreement is at quadrature level.
    assert ours == pytest.approx(theirs, rel=1e-9)


# Tolerances are recorded per case, not globally. The first three are at the level of
# the quadrature itself. The last is not: a `ripple` of 1e-5 sits close to the lower
# bound of its `Interval(0, 1)` constraint, and a `Param` stores it as 1.000010000100001e-05,
# a relative shift of 1e-5 in the coefficient before any profile code runs. Reproduced
# with a bare `prf.Param(value=1e-5, constraint=Interval(0.0, 1.0))`, so it is a
# parameter-layer artefact shared by every `Interval`-constrained parameter near a
# bound, not a Klopfenstein one. The quadrature at that `A` was checked separately
# against tight-tolerance adaptive quadrature and agrees to 4e-16.
@pytest.mark.parametrize('ripple, rel', [(0.5, 1e-12), (0.05, 1e-12), (1e-3, 1e-12), (1e-5, 1e-6)])
def test_klopfenstein_quadrature_holds_across_the_ripple_range(ripple, rel):
    """A small ripple means a large `A`, a more sharply peaked integrand, and the
    hardest case for a fixed-order rule. The rule is claimed accurate across the usable
    range, so the range is tested, not just the default."""
    start, end = 50.0, 100.0
    ts = np.linspace(0.05, 0.95, 11)

    ours = np.asarray(KlopfensteinProfile(start=start, end=end, ripple=ripple).evaluate(jnp.asarray(ts)))
    assert ours == pytest.approx(_skrf_klopfenstein(start, end, ripple, ts), rel=rel)


def test_klopfenstein_endpoint_steps_match_the_published_result():
    r"""Klopfenstein's taper does not reach its terminating impedances: it steps by
    $\Gamma_m = \rho\,\Gamma_0$ at each end. This is the published identity
    $A^2\phi(1, A) = \cosh A - 1$, and it is the classic signature of the design."""
    start, end, ripple = 50.0, 100.0, 0.05
    prof = KlopfensteinProfile(start=start, end=end, ripple=ripple)

    gamma_0 = 0.5 * np.log(end / start)
    gamma_m = ripple * gamma_0

    assert float(prof.evaluate(0.0)) == pytest.approx(start * np.exp(gamma_m), rel=1e-10)
    assert float(prof.evaluate(1.0)) == pytest.approx(end * np.exp(-gamma_m), rel=1e-10)


def test_klopfenstein_matches_pozar_worked_example():
    """Pozar, *Microwave Engineering* (4th ed.), Example 5.8: a Klopfenstein taper
    from 50 to 100 ohm with a maximum passband reflection of $\\Gamma_m = 0.02$.

    The published endpoint identity $A^2\\phi(1, A) = \\cosh A - 1$ fixes the taper's
    value at each end, so $Z(0) = 50 e^{\\Gamma_m}$ and $Z(L) = 100 e^{-\\Gamma_m}$.
    """
    start, end, gamma_m = 50.0, 100.0, 0.02
    gamma_0 = 0.5 * np.log(end / start)
    assert gamma_0 == pytest.approx(0.347, abs=5e-4)

    prof = KlopfensteinProfile(start=start, end=end, ripple=gamma_m / gamma_0)

    assert float(prof.evaluate(0.0)) == pytest.approx(start * np.exp(gamma_m), rel=1e-9)
    assert float(prof.evaluate(1.0)) == pytest.approx(end * np.exp(-gamma_m), rel=1e-9)
    # Symmetric about the geometric mean, as the design's odd phi function requires.
    assert float(prof.evaluate(0.5)) == pytest.approx(np.sqrt(start * end), rel=1e-12)


def test_klopfenstein_is_symmetric_about_its_midpoint():
    r"""$\phi$ is odd, so $\ln Z$ is antisymmetric about $\tfrac12\ln(Z_0 Z_L)$."""
    prof = KlopfensteinProfile(start=50.0, end=100.0, ripple=0.05)
    ts = jnp.linspace(0.0, 1.0, 21)
    values = np.asarray(prof.evaluate(ts))
    assert values * values[::-1] == pytest.approx(50.0 * 100.0, rel=1e-10)


def test_klopfenstein_degenerates_to_an_abrupt_step_at_full_ripple():
    r"""At $\rho = 1$ the design gives up all its taper: $A = 0$, the correction term
    vanishes, and the profile is the constant $\sqrt{Z_0 Z_L}$ -- the abrupt step the
    taper was replacing.

    The `ripple` constraint excludes this bound, but a *fixed* coefficient is not
    range-checked, so the value is reachable. The integrand is $0/0$ there, and this is
    the guard on the quadrature's division: the limit must come out finite, not `nan`.
    """
    start, end = 50.0, 100.0
    ts = jnp.linspace(0.0, 1.0, 11)

    values = KlopfensteinProfile(start=start, end=end, ripple=1.0).evaluate(ts)
    assert jnp.all(jnp.isfinite(values))
    assert values == pytest.approx(np.full(11, np.sqrt(start * end)), rel=1e-12)


# ---------------------------------------------------------
# Smoothness, which later extrapolation depends on
# ---------------------------------------------------------

@pytest.mark.parametrize('cls', [LinearProfile, ExponentialProfile, KlopfensteinProfile])
def test_shipped_profiles_are_c2_on_the_unit_interval(cls):
    """ADR-0004's O(h^4) Richardson claim needs the profile to be C2 in `t`. Every
    shipped shape is, endpoints included, so the second derivative stays finite and
    continuous across [0, 1]."""
    prof = cls(start=50.0, end=100.0)
    d2 = jax.vmap(jax.grad(jax.grad(prof.evaluate)))(jnp.linspace(0.0, 1.0, 101))

    assert jnp.all(jnp.isfinite(d2))
    # Continuity: no step between neighbouring samples large against the scale of
    # the second derivative itself.
    assert float(jnp.max(jnp.abs(jnp.diff(d2)))) < 0.05 * float(jnp.max(jnp.abs(d2)) + 1.0)
