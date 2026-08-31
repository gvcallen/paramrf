# tests/test_optimize/test_fit.py
import pytest
import jax.numpy as jnp
import numpy as np

from pmrf.frequency import Frequency
from pmrf.models import CoaxialLine
from pmrf.materials import BulkConductor, ConstantDielectric
from pmrf.fitting import fit_minimize
from pmrf.parameters import Fixed, Bounded

@pytest.fixture
def fit_freq():
    return Frequency(start=1.0, stop=5.0, npoints=21, unit='GHz')

@pytest.fixture
def truth_model():
    return CoaxialLine(
        din=1.12e-3, 
        dout=3.2e-3, 
        dielectric=ConstantDielectric(epr=1.384, tand=0.001),
        conductor=BulkConductor(rho=1.6e-8),
        length=0.1,  # This is the target length we want to find (10 cm)
        mur=1.0
    )

@pytest.fixture
def target_network(fit_freq, truth_model):
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
        dielectric=ConstantDielectric(epr=Fixed(1.384), tand=Fixed(0.001)),
        conductor=BulkConductor(rho=Fixed(1.6e-8)),
        mur=Fixed(1.0),
        # Start at 9.5 cm to stay within a fraction of a wavelength of 10 cm
        length=Bounded(0.05, 0.15, value=0.095)
    )

def test_fit_skrf_synthetic_data(starting_model, target_network):
    # Note that this also tests full two-port fitting (all S-params).
    # We test only one feature (s21) below
    results = fit_minimize(starting_model, target_network)
    fitted_model = results.model

    assert jnp.allclose(fitted_model.length.value, 0.1, atol=1e-3)

    target_freq = Frequency.from_skrf(target_network.frequency)
    residuals = target_network.s - fitted_model.s(target_freq)
    # Add a tiny epsilon to avoid log10(0) if the fit is mathematically perfect
    max_residual_db = np.max(20 * np.log10(np.abs(residuals) + 1e-15))
    
    assert max_residual_db < -30

def test_fit_raw_ndarray(truth_model, starting_model, fit_freq):
    target_s = np.array(truth_model.s(fit_freq))
    results = fit_minimize(starting_model, target_s, frequency=fit_freq)

    assert jnp.allclose(results.model.length.value, 0.1, atol=1e-3)

def test_fit_missing_freq_error(starting_model, fit_freq):
    dummy_s = jnp.zeros((fit_freq.npoints, 2, 2), dtype=complex)
    with pytest.raises(Exception, match="Frequency must be passed if Network data is not provided"):
        fit_minimize(starting_model, dummy_s, frequency=None)

def test_fit_specific_feature(truth_model, starting_model, fit_freq):
    from pmrf.evaluators import Feature
    s21_mag_target = Feature('s21_mag')(truth_model, fit_freq)
    results = fit_minimize(
        starting_model, 
        s21_mag_target, 
        frequency=fit_freq, 
        features='s21_mag'
    )
    
    assert jnp.allclose(results.model.length.value, 0.1, atol=1e-3)