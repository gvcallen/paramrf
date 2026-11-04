import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.parameters.constraints import Positive
from pmrf.stats.distributions import Uniform, LogUniform, Normal, LogNormal, TruncatedNormal
from pmrf.models import Resistor

from tests._jit import assert_same_jit_key


class RC(prf.Model):
    R: prf.Param = prf.param()
    C: prf.Param = prf.param(scale=1e-12)

    def s(self, freq):
        _TRACES.append(None)
        w = freq.w
        z = self.R + 1 / (1j * w * self.C)
        g = (z - 50.0) / (z + 50.0)
        return g[:, None, None]


_TRACES = []


@pytest.mark.parametrize("factory, distribution", [
    (lambda **kwargs: prf.uniform(1.0, 3.0, **kwargs), Uniform(1.0, 3.0)),
    (lambda **kwargs: prf.loguniform(1.0, 3.0, **kwargs), LogUniform(1.0, 3.0)),
    (lambda **kwargs: prf.normal(2.0, 0.5, **kwargs), Normal(2.0, 0.5)),
    (lambda **kwargs: prf.lognormal(0.0, 0.5, **kwargs), LogNormal(0.0, 0.5)),
    (lambda **kwargs: prf.truncated_normal(2.0, 0.5, 1.0, 3.0, **kwargs),
     TruncatedNormal(2.0, 0.5, 1.0, 3.0)),
])
def test_distribution_parameter_shortcuts_match_random(factory, distribution):
    options = dict(value=2.0, fixed=True, name="p", scale=1e-12,
                   metadata={"source": "test"})
    shortcut = factory(**options)
    reference = prf.random(distribution, **options)
    assert np.allclose(shortcut.value, reference.value)
    assert shortcut.bounds is reference.bounds is None or np.allclose(shortcut.bounds, reference.bounds)
    assert np.allclose(shortcut.physical_value, reference.physical_value)
    assert shortcut.fixed == reference.fixed
    assert shortcut.name == reference.name
    assert shortcut.metadata == reference.metadata
    assert np.allclose(prf.unwrap(prf.update(shortcut, fixed=False).distribution).log_prob(2.0),
                       prf.unwrap(prf.update(reference, fixed=False).distribution).log_prob(2.0))


@pytest.mark.parametrize("factory, distribution", [
    (lambda: prf.uniform(1.0, 3.0), Uniform(1.0, 3.0)),
    (lambda: prf.normal(2.0, 0.5), Normal(2.0, 0.5)),
    (lambda: prf.lognormal(0.0, 0.5), LogNormal(0.0, 0.5)),
    (lambda: prf.truncated_normal(2.0, 0.5, 1.0, 3.0),
     TruncatedNormal(2.0, 0.5, 1.0, 3.0)),
])
def test_distribution_parameter_shortcut_infers_value(factory, distribution):
    assert np.allclose(factory().value, prf.random(distribution).value)


def test_loguniform_factory_default_and_scaled_parameter():
    p = prf.loguniform(1.0, 100.0, scale=1e-3, name="length",
                       metadata={"unit": "mm"})
    assert p.value == pytest.approx(10.0)
    assert p.physical_value == pytest.approx(0.01)
    np.testing.assert_allclose(p.bounds, [1, 100])
    assert p.name == "length"
    assert p.metadata == {"unit": "mm"}
    assert isinstance(prf.unwrap(p.distribution), LogUniform)
    assert prf.loguniform is prf.parameters.loguniform


def test_loguniform_factory_constraint_preserves_log_density():
    constraint = prf.parameters.constraints.Interval(10.0, 100.0)
    p = prf.loguniform(1.0, 1000.0, constraint=constraint, value=20.0,
                       fixed=True, name="gain")
    prior = prf.unwrap(prf.update(p, fixed=False).distribution)
    assert isinstance(prior, LogUniform)
    np.testing.assert_allclose(prior.support, [10, 100])
    assert prior.prob(20.0) == pytest.approx(1 / (20 * np.log(10)))
    assert p.fixed and p.name == "gain" and p.value == pytest.approx(20.0)


def test_distribution_parameter_shortcut_preserves_validation_errors():
    with pytest.raises(Exception, match="outside"):
        prf.uniform(1.0, 3.0, value=4.0)
    with pytest.raises(Exception, match="outside"):
        prf.normal(2.0, 0.5, constraint=prf.parameters.constraints.Interval(1.0, 3.0), value=4.0)


def test_normal_parameter_shortcut_truncates_prior():
    constraint = prf.parameters.constraints.Interval(1.0, 3.0)
    actual = prf.normal(2.0, 0.5, constraint=constraint, value=2.0)
    expected = prf.random(Normal(2.0, 0.5), constraint=constraint, value=2.0)
    assert np.allclose(actual.bounds, expected.bounds)
    assert np.allclose(prf.unwrap(actual.distribution).log_prob(2.0),
                       prf.unwrap(expected.distribution).log_prob(2.0))


def test_parameter_factory_exports_and_prior_distinction():
    for name in ("fixed", "unconstrained", "constrained", "bounded", "random",
                 "uniform", "loguniform", "normal", "lognormal", "truncated_normal"):
        assert getattr(prf, name) is getattr(prf.parameters, name)
    for name in ("Fixed", "Unconstrained", "Constrained", "Bounded", "Random"):
        assert not hasattr(prf, name)
        assert not hasattr(prf.parameters, name)
    assert prf.bounded(1.0, 3.0).distribution is None
    assert prf.uniform(1.0, 3.0).distribution is not None


def test_distribution_parameter_shortcuts_use_sigma_for_spread_and_scale_for_units():
    for actual, reference in (
        (prf.normal(loc=2.0, sigma=0.5, scale=1e-12),
         prf.random(Normal(2.0, 0.5), scale=1e-12)),
        (prf.lognormal(loc=0.0, sigma=0.5, scale=1e-12),
         prf.random(LogNormal(0.0, 0.5), scale=1e-12)),
        (prf.truncated_normal(loc=2.0, sigma=0.5, low=1.0, high=3.0, scale=1e-12),
         prf.random(TruncatedNormal(2.0, 0.5, 1.0, 3.0), scale=1e-12)),
    ):
        assert np.allclose(actual.value, reference.value)
        assert np.allclose(actual.physical_value, reference.physical_value)


class BoundedPF(prf.Model):
    C: prf.Param = prf.param(constraint=Positive(), scale=1e-12)

    def s(self, freq):
        return jnp.zeros((freq.npoints, 1, 1), dtype=complex)


# Value spaces

def test_param_value_spaces():
    p = prf.constrained(Positive(), 2.0, scale=1e-12)
    assert np.allclose(p.value, 2.0)
    assert np.allclose(p.physical_value, 2e-12)
    assert np.allclose(p.raw_to_declared_bijector.forward(p.raw_value), 2.0)
    assert np.allclose(p.declared_to_physical_bijector.forward(p.value), 2e-12)
    assert not hasattr(p, "unscaled_value")


def test_param_variable_field():
    p = prf.unconstrained(2.0)
    assert hasattr(p, "variable") and not isinstance(p.variable, jax.Array)
    q = prf.Param(variable=p.variable, scale=1e-3)
    assert np.allclose(q.value, 2.0)


def test_unwrap_is_physical():
    rc = RC(R=1.0, C=2.0)
    unwrapped = prf.unwrap(rc)
    assert np.allclose(unwrapped.C, 2e-12)
    assert np.allclose(rc.C * 1.0, 2e-12)


# Scale is units

def test_field_scale_is_inherited():
    rc = RC(R=1.0, C=prf.unconstrained(2.0))
    assert np.allclose(rc.C.value, 2.0)
    assert np.allclose(rc.C.physical_value, 2e-12)


def test_explicit_scale_overrides_field():
    rc = RC(R=1.0, C=prf.unconstrained(2.0, scale=1e-9))
    assert np.allclose(rc.C.value, 2.0)
    assert np.allclose(rc.C.physical_value, 2e-9)


def test_bounds_are_declared_space():
    m = BoundedPF(C=prf.bounded(1.0, 3.0, value=2.0))
    assert np.allclose(m.C.physical_value, 2e-12)
    assert np.allclose(m.C.bounds, (1.0, 3.0))
    assert np.allclose(prf.replace(m.C, value=2.5).physical_value, 2.5e-12)
    with pytest.raises(Exception, match="outside the constraint"):
        prf.replace(m.C, value=4.0)


def test_distribution_is_declared_space():
    from pmrf.parameters import node_distribution

    m = BoundedPF(C=prf.random(Uniform(1.0, 3.0), value=2.0))
    assert np.allclose(prf.unwrap(m.C.distribution).mean(), 2.0)
    prior = node_distribution(m.C)
    assert np.isfinite(prior.log_prob(m.C.physical_value))
    assert np.isneginf(prior.log_prob(m.C.value))


# Existing value surfaces follow declared space

def test_values_and_update_are_declared():
    rc = RC(R=1.0, C=2.0)
    assert np.allclose(prf.values(rc)["C"], 2.0)
    updated = prf.update(rc, {"C": 3.0})
    assert np.allclose(updated.C.value, 3.0)
    assert np.allclose(updated.C.physical_value, 3e-12)


def test_replace_value_keeps_everything_else():
    p = prf.random(Uniform(45.0, 55.0), value=50.0, name="R", scale=2.0, metadata={"a": 1})
    q = prf.replace(p, value=51.0)

    assert np.allclose(q.value, 51.0)
    assert np.allclose(q.physical_value, 102.0)
    assert q.name == "R" and q.scale == 2.0 and q.metadata == {"a": 1}
    assert q.distribution == p.distribution
    assert q.bounds is not None and np.allclose(q.bounds, p.bounds)
    assert not q.fixed


def test_replace_value_on_tree():
    load = Resistor(R=prf.random(Uniform(45.0, 55.0), value=50.0), name="load")
    updated = prf.update(load, "R", fn=lambda p: prf.replace(p, value=51.0))
    assert np.allclose(prf.values(updated)["R"], 51.0)


def test_replace_value_keeps_fixed_prior():
    p = prf.random(Uniform(45.0, 55.0), value=50.0, fixed=True)
    q = prf.replace(p, value=51.0)
    assert q.fixed
    assert np.allclose(q.value, 51.0)
    assert np.allclose(prf.update(q, fixed=False).value, 51.0)
    assert prf.update(q, fixed=False).distribution is not None


def test_replace_value_out_of_bounds_raises():
    p = prf.bounded(0.0, 1.0, value=0.5)
    with pytest.raises(Exception, match="outside the constraint"):
        prf.replace(p, value=2.0)


def test_replace_value_is_identity():
    p = prf.random(Uniform(45.0, 55.0), value=50.0, scale=1e-3)
    q = prf.replace(p, value=p.value)
    assert np.allclose(q.value, 50.0)
    assert np.allclose(q.physical_value, 50e-3)


def test_update_out_of_bounds_raises_under_jit():
    import equinox as eqx

    load = Resistor(R=prf.random(Uniform(45.0, 55.0), value=50.0), name="load")

    @eqx.filter_jit
    def set_value(model, v):
        return prf.values(prf.update(model, {"R": v}))["R"]

    assert np.allclose(set_value(load, 51.0), 51.0)
    with pytest.raises(Exception, match="outside the constraint"):
        set_value(load, 60.0)


def test_update_nan_poisons_only_the_invalid_parameter_leaf():
    model = RC(R=prf.constrained(Positive(), 2.0), C=3.0)

    moved = prf.update(model, {"R": -1.0, "C": 4.0}, on_invalid="nan")

    assert np.isnan(moved.R.value)
    assert moved.C.value == 4.0
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(model, {"R": -1.0})


def test_update_nan_handles_open_edges_and_nonfinite_values():
    positive = prf.constrained(Positive(), 1.0)
    bounded = prf.bounded(0.0, 1.0, value=0.5)

    assert np.isnan(prf.update(positive, value=0.0, on_invalid="nan").value)
    assert prf.update(bounded, value=0.0, on_invalid="nan").value == 0.0
    assert prf.update(bounded, value=1.0, on_invalid="nan").value == 1.0
    assert np.isnan(prf.update(bounded, value=2.0, on_invalid="nan").value)
    assert np.isnan(prf.update(prf.unconstrained(1.0), value=jnp.inf, on_invalid="nan").value)
    with pytest.raises(Exception, match="nonfinite"):
        prf.update(prf.unconstrained(1.0), value=jnp.inf)


def test_update_nan_supports_selector_and_raw_value_updates():
    model = RC(R=prf.constrained(Positive(), 2.0), C=3.0)
    selected = prf.update(model, "R", value=0.0, on_invalid="nan")
    raw = prf.update(model.R, value=jnp.inf, space="raw", on_invalid="nan")

    assert np.isnan(selected.R.value)
    assert np.isnan(raw.value)


def test_raw_updates_check_values_after_the_stored_dtype_cast():
    leaf = jnp.array(1.0, dtype=jnp.float32)
    candidate = jnp.array(1e100, dtype=jnp.float64)

    tree = {"x": leaf}
    with pytest.raises(Exception, match="nonfinite"):
        prf.update(tree, {"x": candidate}, space="raw")
    assert np.isnan(prf.update(tree, {"x": candidate}, space="raw", on_invalid="nan")["x"])


def test_update_nan_preserves_array_leaf_shape_and_vmap_trials_are_independent():
    p = prf.constrained(Positive(), jnp.array([1.0, 2.0]))

    def trial(value):
        return prf.update(p, value=value, on_invalid="nan").value

    values = jax.vmap(trial)(jnp.array([[3.0, 4.0], [-1.0, 5.0]]))
    np.testing.assert_allclose(values[0], [3.0, 4.0])
    assert np.isnan(values[1]).all()
    assert values.shape == (2, 2)


def test_update_nan_works_under_filter_jit():
    import equinox as eqx

    p = prf.constrained(Positive(), 1.0)
    update = eqx.filter_jit(lambda value: prf.update(p, value=value, on_invalid="nan").value)

    assert update(2.0) == 2.0
    assert np.isnan(update(-1.0))


def test_update_nan_keeps_valid_results_and_gradients_unchanged():
    p = prf.constrained(Positive(), 1.0)
    strict = prf.update(p, value=2.0)
    tolerant = prf.update(p, value=2.0, on_invalid="nan")
    model = RC(R=p, C=3.0)
    frequency = prf.Frequency(1.0, 2.0, 3, "GHz")

    assert tolerant.value == strict.value
    assert jax.grad(lambda x: prf.update(p, value=x, on_invalid="nan").value**2)(2.0) == 4.0
    strict_objective = jnp.sum(jnp.abs(prf.update(model, {"R": 2.0}).s(frequency)) ** 2)
    tolerant_objective = jnp.sum(jnp.abs(prf.update(model, {"R": 2.0}, on_invalid="nan").s(frequency)) ** 2)
    assert tolerant_objective == strict_objective


def test_update_nan_rejects_other_modes_and_preserves_dtype():
    p = prf.constrained(Positive(), 2.0)
    valid = prf.update(p, value=3, on_invalid="nan")
    invalid = prf.update(p, value=-1, on_invalid="nan")
    assert valid.value.dtype == p.value.dtype
    assert np.isnan(invalid.value)
    assert_same_jit_key(p, valid)
    with pytest.raises(TypeError, match="only to numerical value updates"):
        prf.update(p, fixed=True, on_invalid="nan")
    with pytest.raises(ValueError, match="must be 'raise' or 'nan'"):
        prf.update(p, value=3, on_invalid="ignore")
    with pytest.raises(TypeError, match="floating-point or complex"):
        prf.update({"count": jnp.array(2, dtype=jnp.int32)}, {"count": 3}, on_invalid="nan")


def test_param_constructor_out_of_bounds_raises_under_jit():
    import equinox as eqx

    @eqx.filter_jit
    def build(v):
        return prf.bounded(0.0, 1.0, value=v).value

    with pytest.raises(Exception, match="outside the constraint"):
        build(2.0)


# Value changes keep the jit cache key

_PARAMS = [
    prf.unconstrained(2.0),
    prf.unconstrained(jnp.asarray([1.0, 2.0])),
    prf.constrained(Positive(), 2.0, scale=1e-12),
    prf.bounded(0.0, 5.0, value=2.0),
    prf.random(Uniform(0.0, 5.0), value=2.0, fixed=True),
    prf.fixed(2.0),
]
_NEW_VALUES = [3.0, np.float64(3.0), jnp.asarray(3.0), jnp.asarray(3, dtype=jnp.int32)]


@pytest.mark.parametrize("p", _PARAMS)
@pytest.mark.parametrize("v", _NEW_VALUES)
def test_replace_keeps_jit_key(p, v):
    v = jnp.broadcast_to(v, jnp.shape(p.value))
    assert_same_jit_key(p, prf.replace(p, value=v))


@pytest.mark.parametrize("v", _NEW_VALUES)
def test_update_keeps_jit_key(v):
    rc = RC(R=1.0, C=prf.bounded(0.0, 5.0, value=2.0))
    assert_same_jit_key(rc, prf.update(rc, {"R": v, "C": v}))


def test_value_change_does_not_recompile():
    freq = prf.Frequency(1, 2, 3, "GHz")
    rc = RC(R=1.0, C=2.0)
    _TRACES.clear()
    prf.update(rc, {"C": 3.0}).s(freq)
    assert len(_TRACES) == 1
    prf.update(rc, {"C": jnp.asarray(4.0)}).s(freq)
    prf.update(rc, {"R": np.float64(5.0)}).s(freq)
    assert len(_TRACES) == 1
