# tests/test_optimize/test_fit.py
import pytest
import jax.numpy as jnp
import numpy as np

from pmrf.frequency import Frequency
from pmrf.models import CoaxialLine
from pmrf.fitting import fit_minimize
from pmrf.parameters import Fixed, Bounded

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def fit_freq():
    # A standard frequency sweep for the cable
    return Frequency(start=1.0, stop=5.0, npoints=21, unit='GHz')

@pytest.fixture
def truth_model():
    """The 'Golden' model representing our synthetic measured data."""
    return CoaxialLine(
        din=1.12e-3, 
        dout=3.2e-3, 
        epr=1.384, 
        rho=1.6e-8, 
        tand=0.001, 
        length=0.1,  # This is the target length we want to find (10 cm)
        mur=1.0
    )

@pytest.fixture
def target_network(fit_freq, truth_model):
    """Generates an in-memory scikit-rf Network of the truth_model."""
    skrf = pytest.importorskip("skrf")
    s_target = np.array(truth_model.s(fit_freq))
    freq_skrf = fit_freq.to_skrf()
    
    return skrf.Network(frequency=freq_skrf, s=s_target, z0=50)

@pytest.fixture
def starting_model():
    """
    The model we will actually optimize. 
    """
    return CoaxialLine(
        din=Fixed(1.12e-3),
        dout=Fixed(3.2e-3),
        epr=Fixed(1.384),
        rho=Fixed(1.6e-8),
        tand=Fixed(0.001),
        mur=Fixed(1.0),
        # Start at 9.5 cm to stay within a fraction of a wavelength of 10 cm!
        length=Bounded(0.05, 0.15, value=0.095)
    )

# ---------------------------------------------------------
# Fitting Tests
# ---------------------------------------------------------

# def test_fit_skrf_synthetic_data(starting_model, target_network):
#     """
#     Replaces the old file-loading test.
#     Fits a perturbed model to an in-memory scikit-rf Network.
#     """
#     # Notice we don't pass frequency; `fit` should extract it from the Network
#     results = fit_minimize(starting_model, target_network)
#     fitted_model = results.model

#     # 1. Did it find the correct physical length?
#     assert jnp.allclose(fitted_model.length.value, 0.1, atol=1e-3)

#     # 2. Check the residuals just like the old test
#     import pmrf as prf
#     target_freq = prf.Frequency.from_skrf(target_network.frequency)
    
#     residuals = target_network.s - fitted_model.s(target_freq)
#     # Add a tiny epsilon to avoid log10(0) if the fit is mathematically perfect
#     max_residual_db = np.max(20 * np.log10(np.abs(residuals) + 1e-15))
    
#     assert max_residual_db < -30

def test_fit_raw_ndarray(truth_model, starting_model, fit_freq):
    """
    Ensure the fit wrapper handles raw JAX arrays correctly.
    """
    # Extract raw S-parameter array
    target_s = np.array(truth_model.s(fit_freq))

    # Must pass frequency explicitly when using raw arrays
    results = fit_minimize(starting_model, target_s, frequency=fit_freq)
    
    assert jnp.allclose(results.model.length.value, 0.1, atol=1e-3)

# def test_fit_missing_freq_error(starting_model, fit_freq):
#     """
#     Ensure `fit` throws an error if an ndarray is provided without a Frequency axis.
#     """
#     dummy_s = jnp.zeros((fit_freq.npoints, 2, 2), dtype=complex)
    
#     with pytest.raises(Exception, match="Frequency must be passed if Network data is not provided"):
#         fit_minimize(starting_model, dummy_s, frequency=None)

# def test_fit_specific_feature(truth_model, starting_model, fit_freq):
#     """
#     Ensure the `features` argument correctly maps strings to evaluators.
#     Instead of fitting the full complex S-matrix, we fit ONLY the S21 magnitude.
#     """
#     # Use the Feature evaluator manually to get our target data
#     from pmrf.evaluators import Feature
#     s21_mag_target = Feature('s21_mag')(truth_model, fit_freq)

#     # Pass the string alias into the fit wrapper
#     results = fit_minimize(
#         starting_model, 
#         s21_mag_target, 
#         frequency=fit_freq, 
#         features='s21_mag'
#     )
    
#     # Even fitting just the scalar magnitude should walk the length back to 0.1
#     assert jnp.allclose(results.model.length.value, 0.1, atol=1e-3)