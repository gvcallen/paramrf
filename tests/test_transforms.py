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
        prf.values(model, space=space)
    )

    actual = prf.values(d_model)
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
    declared, physical = prf.values(declared), prf.values(physical)

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
    r_fixed, r_free = prf.values(d_fixed)['r.R'], prf.values(d_free)['r.R']

    assert r_free != 0
    assert jnp.allclose(r_fixed, r_free, rtol=1e-12, atol=0)


def test_derivative_through_a_tie_reaches_the_source():
    freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    tied = prf.tie(_filter(), 'r.R', 'c1.C', fn=lambda C: C * 5.0)

    def s21(m):
        return m.s_mag(freq)[0, 1, 0]

    (d_tied,) = prf.derivative(s21, tied)
    expected = jax.grad(lambda v: s21(prf.update(tied, v)))(prf.values(tied))

    actual = prf.values(d_tied)
    assert actual.keys() == {'c1.C', 'l1.L'}
    for name in expected:
        assert jnp.allclose(actual[name], expected[name], rtol=1e-12, atol=0), name


def test_derivative_jacobian_is_per_declared_unit():
    band = prf.Frequency(1, 5, 11, 'GHz')
    model = _filter()

    (d_model,) = prf.derivative(lambda m: m.s_mag(band)[:, 1, 0], model)
    (physical,) = prf.derivative(lambda m: m.s_mag(band)[:, 1, 0], model, space='physical')

    declared = prf.values(d_model)
    assert declared['c1.C'].shape == (11,)
    assert jnp.allclose(declared['c1.C'], prf.values(physical)['c1.C'] * 1e-12, rtol=1e-12, atol=0)


@pytest.fixture
def autodiff_calls(monkeypatch):
    """Records which of `jax.grad`, `jax.jacfwd` and `jax.jacrev` were called."""
    calls = []
    for name in ('grad', 'jacfwd', 'jacrev'):
        original = getattr(jax, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(jax, name, spy)
    return calls


def test_derivative_of_a_wide_real_output_uses_forward_mode(autodiff_calls):
    freq = prf.Frequency(1, 1000, 1000, 'MHz')
    from pmrf.models import ShuntCapacitor
    cap = ShuntCapacitor(C=prf.Unconstrained(1.0, scale=1e-12), name='c1')

    (d_cap,) = prf.derivative(lambda m: m.s_mag(freq)[:, 1, 0], cap)

    assert autodiff_calls == ['jacfwd']
    assert prf.values(d_cap)['C'].shape == (1000,)


def test_derivative_of_a_tall_real_output_uses_reverse_mode(autodiff_calls):
    x = jnp.linspace(0.0, 1.0, 1000)

    (dx,) = prf.derivative(lambda x: jnp.stack([x.sum(), (x ** 2).sum(), x[0]]), x)

    assert autodiff_calls == ['jacrev']
    assert dx.shape == (3, 1000)
    assert jnp.allclose(dx[1], 2 * x)


def test_derivative_of_a_scalar_output_uses_grad(autodiff_calls):
    (dx,) = prf.derivative(lambda x: (x ** 2).sum(), jnp.array([1.0, 2.0]))

    assert autodiff_calls == ['grad']
    assert jnp.allclose(dx, jnp.array([2.0, 4.0]))


def _assert_complex_derivative_is_split_parts(fn, model):
    (d_complex,) = prf.derivative(fn, model)
    (d_re,) = prf.derivative(lambda m: fn(m).real, model)
    (d_im,) = prf.derivative(lambda m: fn(m).imag, model)

    actual, re, im = prf.values(d_complex), prf.values(d_re), prf.values(d_im)
    assert actual.keys() == re.keys()
    for name in actual:
        assert jnp.iscomplexobj(actual[name]), name
        # The same forward-mode operations on both sides; only reduction order can differ.
        assert jnp.allclose(actual[name], re[name] + 1j * im[name], rtol=1e-12, atol=0), name
    return actual


def test_derivative_of_a_wide_complex_output(autodiff_calls):
    from pmrf.models import ShuntCapacitor
    freq = prf.Frequency(1, 1000, 1000, 'MHz')
    cap = ShuntCapacitor(C=prf.Unconstrained(1.0, scale=1e-12), name='c1')

    actual = _assert_complex_derivative_is_split_parts(lambda m: m.s(freq), cap)

    assert actual['C'].shape == (1000, 2, 2)
    assert autodiff_calls[0] == 'jacfwd'


def test_derivative_of_a_tall_complex_output(autodiff_calls):
    from pmrf.models import Inductor, ShuntCapacitor
    freq = prf.Frequency(1, 2, 2, 'GHz')
    model = (
        _filter()
        ** Inductor(L=prf.Unconstrained(2.0, scale=1e-9), name='l2')
        ** ShuntCapacitor(C=prf.Unconstrained(0.5, scale=1e-12), name='c2')
    )

    # 2 complex outputs (4 reals) against 5 parameters.
    actual = _assert_complex_derivative_is_split_parts(lambda m: m.s(freq)[:, 1, 0], model)

    assert len(actual) == 5
    assert actual['c1.C'].shape == (2,)
    assert autodiff_calls[0] == 'jacfwd'


def test_derivative_of_a_complex_scalar_output():
    fn = lambda x: jnp.exp(1j * x).sum()
    x = jnp.array([0.3, 0.7])

    (dx,) = prf.derivative(fn, x)

    assert jnp.allclose(dx, 1j * jnp.exp(1j * x), rtol=1e-12, atol=0)


def test_derivative_rejects_a_complex_input():
    with pytest.raises(TypeError, match="Complex-valued inputs are not supported"):
        prf.derivative(lambda z: jnp.abs(z) ** 2, jnp.array([1.0 + 2.0j]))


def test_derivative_rejects_an_unknown_space():
    with pytest.raises(ValueError, match="Unknown space"):
        prf.derivative(lambda x: x ** 2, jnp.array(1.0), space='unscaled')


@pytest.mark.parametrize("space", ["declared", "physical"])
@pytest.mark.parametrize("k2", [0.0, 1e-6])
def test_derivative_on_and_near_a_closed_bound(space, k2):
    """`k2` is NonNegative: at 0 its raw value is -inf, which must not reach the derivative."""
    from pmrf.models import DatasheetLine

    line = DatasheetLine(length=1.0, zn=50.0, vf=0.8, k1=0.1, k2=k2)

    (d_line,) = prf.derivative(lambda m: 2.0 * m.k2, line, space=space)

    assert prf.values(d_line)['k2'] == 2.0


def test_a_closed_bound_is_still_a_valid_value():
    from pmrf.models import DatasheetLine

    line = DatasheetLine(length=1.0, zn=50.0, vf=0.8, k1=0.1, k2=0.5)

    assert prf.values(prf.update(line, {'k2': 0.0}))['k2'] == 0.0
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(line, {'k2': -1.0})


_CLOSED_BOUNDS = {
    'non_negative': (0.0, prf.parameters.constraints.NonNegative()),
    'greater_than': (2.0, prf.parameters.constraints.GreaterThan(2.0)),
    'interval_upper': (3.0, prf.parameters.constraints.Interval(1.0, 3.0)),
    'less_than': (3.0, prf.parameters.constraints.LessThan(3.0)),
}


@pytest.mark.parametrize("n", [1, 50], ids=['tall', 'wide'])
@pytest.mark.parametrize("space", ["declared", "physical"])
@pytest.mark.parametrize("bound", list(_CLOSED_BOUNDS))
def test_derivative_on_a_closed_bound_is_exact_and_confined(bound, space, n):
    """On a closed bound the derivative is finite and exact, and does not reach the
    other parameters, whether taken by reverse mode (tall) or forward mode (wide)."""
    value, constraint = _CLOSED_BOUNDS[bound]
    scale = 1e-3
    tree = {
        'a': prf.Param(value=value, constraint=constraint, scale=scale),
        'b': prf.Unconstrained(1.5),
    }
    x = jnp.linspace(0.0, 1.0, n)

    (d_tree,) = prf.derivative(lambda t: 2.0 * t['a'] + t['b'] * x, tree, space=space)

    per_unit = 1.0 if space == 'physical' else scale
    assert jnp.array_equal(d_tree['a'], jnp.full(n, 2.0 * per_unit))
    assert jnp.array_equal(d_tree['b'], x)


def test_derivative_in_physical_space_on_a_closed_bound_is_declared_over_scale():
    tree = {'a': prf.Param(value=0.0, constraint=prf.parameters.constraints.NonNegative(), scale=1e-3)}
    fn = lambda t: jnp.sin(t['a'] + 0.3)

    (declared,) = prf.derivative(fn, tree)
    (physical,) = prf.derivative(fn, tree, space='physical')

    assert jnp.isfinite(declared['a'])
    assert jnp.allclose(physical['a'], declared['a'] / 1e-3, rtol=1e-12, atol=0)


def _coax_on_a_closed_bound():
    from pmrf.models import CoaxialLine, SchelkunoffCoaxialFormulation
    from pmrf.materials import DjordjevicSarkarDielectric, BulkConductor

    # The dielectric's conductivity defaults to 0, the closed bound of NonNegative.
    return CoaxialLine(
        length=1.0, d_in=3.124e-3, d_out=8.328e-3,
        dielectric=DjordjevicSarkarDielectric(ep_r=1.33, tand=1e-4, f_low=1.5e4, f_high=1e11, f_ref=1e8),
        conductor=BulkConductor(sigma=5.8e7), formulation=SchelkunoffCoaxialFormulation(),
    )


@pytest.mark.parametrize("n", [5, 50], ids=['tall', 'wide'])
def test_derivative_of_a_line_with_a_parameter_on_a_closed_bound(n):
    """A parameter on a closed bound gives every parameter a finite derivative, and the
    others match a reference taken through free parameters inside their bounds."""
    line = _coax_on_a_closed_bound()
    freq = prf.Frequency(50, 130, n, 'MHz')
    fn = lambda m: jnp.abs(m.s(freq)[:, 1, 1])

    (d_line,) = prf.derivative(fn, line)
    actual = prf.values(d_line)

    assert prf.values(line)['dielectric.sigma'] == 0.0
    assert all(jnp.isfinite(v).all() for v in actual.values()), actual

    # Fixed parameters stop the gradient through `prf.update`, so the reference frees them.
    names = ['length', 'd_in', 'd_out', 'dielectric.ep_r', 'dielectric.tand']
    free = prf.update(line, names, fixed=False)
    values = prf.values(free, names)
    expected = jax.jacfwd(lambda v: fn(prf.unwrap(prf.update(free, v))))(values)
    for name in names:
        assert jnp.allclose(actual[name], expected[name], rtol=1e-12, atol=0), name


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