import jax.numpy as jnp

import pmrf as prf
from pmrf.models import ShuntCapacitor


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
    cap = ShuntCapacitor(C=jnp.array(1.0e-12), name='test_cap')

    def eval_s21(model):
        return model.s_mag(freq)[0, 1, 0]

    (d_cap,) = prf.derivative(eval_s21, cap)

    # The returned derivative should be structurally identical to the input model
    assert isinstance(d_cap, type(cap))
    assert hasattr(d_cap, "C")
    assert d_cap.C is not None 

    # Name is static so should be unaffected
    assert d_cap.name == 'test_cap'


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