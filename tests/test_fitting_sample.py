# tests/test_fitting/test_fit_sample.py

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from pmrf.frequency import Frequency
from pmrf.models import CoaxialLine
from pmrf.fitting import fit_sample
from pmrf.parameters import Fixed, Random
from pmrf.distributions import Normal
from pmrf.infer import NUTS

@pytest.fixture
def fit_freq():
    return Frequency(start=1.0, stop=5.0, npoints=11, unit='GHz')

@pytest.fixture
def truth_model():
    return CoaxialLine(
        din=1.12e-3, 
        dout=3.2e-3, 
        epr=1.384, 
        rho=1.6e-8, 
        tand=0.001, 
        length=0.1,  # Target length
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
    return CoaxialLine(
        din=Fixed(1.12e-3),
        dout=Fixed(3.2e-3),
        epr=Fixed(1.384),
        rho=Fixed(1.6e-8),
        tand=Fixed(0.001),
        mur=Fixed(1.0),
        # Normal prior centered at 0.1 with standard deviation 0.05
        length=Random(Normal(0.1, 0.05), value=jnp.array(0.095))
    )

def test_fit_sample_skrf_synthetic_data(starting_model, target_network):
    key = jax.random.key(42)
    solver = NUTS(num_warmup=10)

    results = fit_sample(
        model=starting_model, 
        data=target_network,
        solver=solver,
        key=key,
        max_steps=20
    )

    assert results.data is target_network
    assert results.solution.sampled_model.length.shape == (20,)
    assert results.solution.fn_values.shape == (20,)
    assert results.solution.best_model.length.ndim == 0

def test_fit_sample_raw_ndarray(truth_model, starting_model, fit_freq):
    # Extract raw S-parameter array
    target_s = np.array(truth_model.s(fit_freq))
    key = jax.random.key(42)
    solver = NUTS(num_warmup=5)

    # Must pass frequency explicitly when using raw arrays
    results = fit_sample(
        model=starting_model, 
        data=target_s, 
        frequency=fit_freq,
        solver=solver,
        key=key,
        max_steps=10
    )
    
    assert results.frequency is fit_freq
    assert results.solution.sampled_model.length.shape == (10,)

def test_fit_sample_missing_freq_error(starting_model, fit_freq):
    dummy_s = jnp.zeros((fit_freq.npoints, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="Frequency must be passed if Network data is not provided"):
        fit_sample(starting_model, dummy_s, frequency=None)

def test_fit_sample_specific_feature(truth_model, starting_model, fit_freq):
    from pmrf.evaluators import Feature
    s21_mag_target = Feature('s21_mag')(truth_model, fit_freq)
    
    key = jax.random.key(42)
    solver = NUTS(num_warmup=5)

    # Pass the string alias into the fit wrapper
    results = fit_sample(
        model=starting_model, 
        data=s21_mag_target, 
        frequency=fit_freq, 
        features='s21_mag',
        solver=solver,
        key=key,
        max_steps=10
    )
    
    # Asserting successful graph evaluation
    assert results.solution.sampled_model.length.shape == (10,)