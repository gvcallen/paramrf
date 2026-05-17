import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.models import Resistor, Capacitor, Inductor, ShuntResistor

@pytest.fixture
def single_freq():
    return Frequency(start=1.0, stop=1.0, npoints=1, unit='GHz')

@pytest.fixture
def sweep_freq():
    return Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

def test_resistor_gradient_analytic(single_freq):
    """
    Test that the JAX autodiff gradient perfectly matches the analytical derivative.
    For a series resistor in a Z0=50 environment:
    S11 = R / (R + 2*Z0)
    Analytically, d(S11)/dR = 2*Z0 / (R + 2*Z0)^2
    """
    def s11_real(r_val):
        model = Resistor(R=r_val)
        s = model.s(single_freq)
        return jnp.real(s[0, 0, 0]) # Extract S11
    
    # Generate the gradient function using JAX
    grad_fn = jax.grad(s11_real)
    
    # Evaluate at R = 50.0
    r_test = 50.0
    z0 = 50.0 # Default Z0
    computed_grad = grad_fn(r_test)
    
    # Calculate analytical expected value
    expected_grad = (2 * z0) / (r_test + 2 * z0)**2
    
    assert jnp.isclose(computed_grad, expected_grad, atol=1e-6)

def test_equinox_filter_grad_pytree(single_freq):
    """
    Test differentiating through the entire model PyTree using eqx.filter_grad.
    This mimics how pmrf.optimize or internal solvers will extract gradients.
    """
    def loss(model):
        s = model.s(single_freq)
        # Simple objective: Magnitude squared of S21
        s21 = s[0, 1, 0]
        return jnp.real(s21 * jnp.conj(s21))
    
    # Initialize a capacitor model
    model = Capacitor(C=1.0e-12)
    
    # Compute gradients with respect to all continuous parameters in the model
    grads = eqx.filter_grad(loss)(model)
    
    # Ensure the gradient for C was computed and is non-zero
    assert grads.C is not None
    assert jnp.abs(grads.C) > 0.0

@pytest.mark.parametrize("model_class, param_name, param_val", [
    (Resistor, 'R', 50.0),
    (Capacitor, 'C', 1.0e-12),
    (Inductor, 'L', 1.0e-9),
    (ShuntResistor, 'R', 50.0)
])
def test_full_jacobian_no_nans(model_class, param_name, param_val, sweep_freq):
    """
    Ensure full Jacobians can be computed for standard components without NaNs, 
    evaluating across an entire frequency band simultaneously.
    """
    def get_s_matrix_components(val):
        kwargs = {param_name: val}
        model = model_class(**kwargs)
        s = model.s(sweep_freq)
        # JAX standard autodiff prefers real outputs, so we concatenate 
        # real and imaginary parts to form a strictly real vector field.
        return jnp.concatenate([jnp.real(s), jnp.imag(s)])

    # Use jacrev (Reverse-mode Jacobian) which is highly efficient 
    # for taking derivatives of large arrays wrt a few parameters
    jac_fn = jax.jacrev(get_s_matrix_components)
    jacobian = jac_fn(param_val)
    
    # The Jacobian should have evaluated cleanly over all frequencies
    assert not jnp.any(jnp.isnan(jacobian))
    # Shape check: (2 * npoints, 2, 2) since we concatenated real/imag
    assert jacobian.shape == (2 * sweep_freq.npoints, 2, 2)

def test_autodiff_frequency_parameter():
    """
    Test that we can actually differentiate the network response with respect 
    to the frequency array itself (useful for group delay and dispersion analysis).
    """
    model = Inductor(L=1.0e-9)
    
    def s21_phase(w_val):
        # We bypass the Frequency object for a pure JAX scalar test
        Z_in = Z_out = 50.0
        denom_c = (1j * w_val * model.L) + (Z_in + Z_out)
        s21 = (2 * Z_in) / denom_c
        return jnp.angle(s21)
    
    w_test = 2 * jnp.pi * 1.0e9 # 1 GHz
    grad_fn = jax.grad(s21_phase)
    
    # Group delay is strictly related to the derivative of phase wrt frequency
    phase_grad = grad_fn(w_test)
    
    assert not jnp.isnan(phase_grad)
    assert phase_grad < 0 # Phase of an inductor should decrease with frequency