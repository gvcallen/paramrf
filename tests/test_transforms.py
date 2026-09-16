import pytest

import jax
import equinox as eqx
import jax.numpy as jnp

import pmrf as prf
from pmrf.models import Capacitor


def test_derivative_math_and_static_filtering():
    """
    Verifies that analytical derivatives are calculated correctly 
    and that static arguments (like strings) do not crash the JAX tracer.
    """
    def eval_fn(x, y, static_name):
        # f(x, y) = x^2 + 3y
        # df/dx = 2x, df/dy = 3
        return x ** 2 + 3 * y

    x_nom = jnp.array(2.0)
    y_nom = jnp.array(4.0)
    
    dx, dy, d_name = prf.derivative(eval_fn, x_nom, y_nom, "ignore_me")

    assert jnp.allclose(dx, 4.0)       # 2 * 2.0 = 4.0
    assert jnp.allclose(dy, 3.0)       # Constant derivative of 3y
    assert d_name is None              # Static string should yield no gradient


def test_derivative_model_structure():
    """
    Verifies that when passing an Equinox model (PyTree), the derivative 
    returns a structurally identical PyTree without throwing arbitrary-type errors.
    """
    freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    cap = Capacitor(C=jnp.array(1.0e-12), name='test_cap')

    def eval_s21(model):
        return model.s_mag(freq)[0, 1, 0]

    (d_cap,) = prf.derivative(eval_s21, cap)

    # The returned derivative should be structurally identical to the input model
    assert isinstance(d_cap, type(cap))
    assert hasattr(d_cap, "C")
    assert d_cap.C is not None 

    # Name is static so should be unaffected
    assert d_cap.name == 'test_cap'


def _filter():
    from pmrf.models import Resistor, Inductor, ShuntCapacitor

    c1 = ShuntCapacitor(C=prf.Unconstrained(1.0, scale=1e-12), name='c1')
    l1 = Inductor(L=prf.Bounded(0.5, 2.0, value=1.0, scale=1e-9), name='l1')
    r = Resistor(5.0, name='r')
    return c1 ** l1 ** r


@pytest.mark.parametrize("space", ["declared", "physical", "raw"])
def test_derivative_is_taken_in_the_requested_space(space):
    """A parameter's derivative is with respect to its value in `space`, which is
    what differentiating through `prf.update` in that space gives."""
    freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    model = prf.update(_filter(), 'r.R', fixed=False)

    def s21(m):
        return m.s_mag(freq)[0, 1, 0]

    (d_model,) = prf.derivative(s21, model, space=space)
    expected = jax.grad(lambda v: s21(prf.update(model, v, space=space)))(
        prf.param_values(model, space=space)
    )

    actual = prf.param_values(d_model)
    assert actual.keys() == expected.keys()
    for name in expected:
        # Both sides run the same float64 operations; only reduction order can differ.
        assert jnp.allclose(actual[name], expected[name], rtol=1e-12, atol=0), name


def test_derivative_declared_is_physical_times_scale():
    freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    model = _filter()

    def s21(m):
        return m.s_mag(freq)[0, 1, 0]

    (declared,) = prf.derivative(s21, model)
    (physical,) = prf.derivative(s21, model, space='physical')
    declared, physical = prf.param_values(declared), prf.param_values(physical)

    assert jnp.allclose(declared['c1.C'], physical['c1.C'] * 1e-12, rtol=1e-12, atol=0)
    assert jnp.allclose(declared['l1.L'], physical['l1.L'] * 1e-9, rtol=1e-12, atol=0)


def test_derivative_of_a_fixed_parameter_is_its_sensitivity():
    """Fixed is an optimisation state; the derivative is the same as when free."""
    freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    fixed = _filter()
    free = prf.update(fixed, 'r.R', fixed=False)

    def s21(m):
        return m.s_mag(freq)[0, 1, 0]

    (d_fixed,) = prf.derivative(s21, fixed)
    (d_free,) = prf.derivative(s21, free)
    r_fixed, r_free = prf.param_values(d_fixed)['r.R'], prf.param_values(d_free)['r.R']

    assert r_free != 0
    assert jnp.allclose(r_fixed, r_free, rtol=1e-12, atol=0)


def test_derivative_through_a_tie_reaches_the_source():
    freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    tied = prf.tie(_filter(), 'r.R', 'c1.C', fn=lambda C: C * 5.0)

    def s21(m):
        return m.s_mag(freq)[0, 1, 0]

    (d_tied,) = prf.derivative(s21, tied)
    expected = jax.grad(lambda v: s21(prf.update(tied, v)))(prf.param_values(tied))

    actual = prf.param_values(d_tied)
    assert actual.keys() == {'c1.C', 'l1.L'}
    for name in expected:
        assert jnp.allclose(actual[name], expected[name], rtol=1e-12, atol=0), name


def test_derivative_jacobian_is_per_declared_unit():
    band = prf.Frequency(1, 5, 11, 'GHz')
    model = _filter()

    (d_model,) = prf.derivative(lambda m: m.s_mag(band)[:, 1, 0], model)
    (physical,) = prf.derivative(lambda m: m.s_mag(band)[:, 1, 0], model, space='physical')

    declared = prf.param_values(d_model)
    assert declared['c1.C'].shape == (11,)
    assert jnp.allclose(declared['c1.C'], prf.param_values(physical)['c1.C'] * 1e-12, rtol=1e-12, atol=0)


def test_derivative_rejects_an_unknown_space():
    with pytest.raises(ValueError, match="Unknown space"):
        prf.derivative(lambda x: x ** 2, jnp.array(1.0), space='unscaled')


def test_sweep_parallel():
    """
    Verifies that a standard sweep correctly vectorizes across the leading 
    dimension of multiple dynamic arrays, while safely passing static objects.
    """
    c_vals = jnp.linspace(1e-12, 5e-12, 10)
    l_vals = jnp.linspace(1e-9, 5e-9, 10)

    def eval_dummy(c, l, static_flag):
        # A simple mathematical check: shape should be (10,)
        return c * l

    # Execute parallel sweep
    out = prf.sweep(eval_dummy, c_vals, l_vals, False)

    assert out.shape == (10,)


def test_sweep_grid_shape():
    """
    Verifies that a Cartesian grid sweep correctly combinations inputs 
    and reshapes the output tensor to mirror the dimensional inputs.
    """
    c_vals = jnp.ones(10)  # Length 10
    l_vals = jnp.ones(5)   # Length 5
    r_vals = jnp.ones(3)   # Length 3

    def eval_dummy(c, l, r, static_text):
        return c + l + r

    # Execute grid sweep
    out = prf.sweep(eval_dummy, c_vals, l_vals, r_vals, "static_text", grid=True)

    # The static string is ignored in sizing, and the three dynamic arrays 
    # should create a 3D tensor of shape (10, 5, 3)
    assert out.shape == (10, 5, 3)

class MockBatchedModel(eqx.Module):
    """A minimal PyTree to simulate batched Bayesian outputs."""
    param: jax.Array
    static_name: str

def test_sweep_with_template():
    """
    Verifies that sweep successfully maps over a batched PyTree when 
    provided with a structural unbatched template.
    """
    # Create an unbatched template
    template_model = MockBatchedModel(param=jnp.array(1.0), static_name="fixed")
    
    # Create a batched version (e.g., 5 samples from a posterior)
    batched_model = MockBatchedModel(param=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), static_name="fixed")
    
    def eval_fn(model):
        return model.param * 2.0
        
    # Execute sweep using the template
    out = prf.sweep(eval_fn, batched_model, template=template_model)
    
    assert out.shape == (5,)
    assert jnp.allclose(out, jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]))

def test_sweep_multiple_templates():
    """
    Verifies that sweep can handle multiple arguments requiring templates.
    """
    t1 = MockBatchedModel(param=jnp.array(1.0), static_name="a")
    t2 = MockBatchedModel(param=jnp.array(1.0), static_name="b")
    
    b1 = MockBatchedModel(param=jnp.array([1.0, 2.0]), static_name="a")
    b2 = MockBatchedModel(param=jnp.array([10.0, 20.0]), static_name="b")
    
    def eval_fn(m1, m2):
        return m1.param + m2.param
        
    # Sweep over both batched models
    out = prf.sweep(eval_fn, b1, b2, template=(t1, t2))
    
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([11.0, 22.0]))

def test_sweep_template_validation_errors():
    """
    Verifies that sweep catches misconfigurations when using templates.
    """
    t1 = MockBatchedModel(param=jnp.array(1.0), static_name="a")
    b1 = MockBatchedModel(param=jnp.ones(5), static_name="a")
    
    # Using grid=True with template should fail
    with pytest.raises(ValueError, match="`grid=True` is not supported"):
        prf.sweep(lambda x: x, b1, grid=True, template=t1)
        
    # Passing the wrong number of templates should fail
    with pytest.raises(ValueError, match="Expected 2 templates"):
        prf.sweep(lambda x, y: x, b1, b1, template=t1) # Passing 2 args but only 1 template