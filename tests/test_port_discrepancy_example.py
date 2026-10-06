"""Execute the docs example through its public recipe functions."""
import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'port_discrepancy_example',
    Path(__file__).parents[1] / 'docs/examples/port_discrepancy.py',
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize('reciprocal', [False, True], ids=['full', 'reciprocal'])
def test_log_event_round_trip_and_continuous_phase(reciprocal):
    frequency = example.prf.Frequency(10, 500, 1000, 'MHz')
    nominal = example.line(0.5).s(frequency)
    observed = example.truth(frequency).s(frequency)
    observed = observed.at[:, 0, 1].multiply(np.exp(0.03 + 0.04j))
    observed = observed.at[:, 1, 0].multiply(np.exp(-0.02 + 0.06j))
    if reciprocal:
        nominal = example.reciprocal_predictor(example.line(0.5), frequency)
        observed = example.jnp.stack((observed[:, 0, 0], observed[:, 1, 1], observed[:, 1, 0]), axis=-1)
        transform = example.ReciprocalEventTransform(nominal)
        expected = -2 * np.log(np.abs(observed[:, 2]))
    else:
        transform = example.PortEventTransform(nominal)
        expected = -2 * np.log(np.abs(observed[:, 0, 1])) - 2 * np.log(np.abs(observed[:, 1, 0])) - 2 * np.log(2)
    event, forward_logdet = transform.forward_and_log_det(observed)
    restored, inverse_logdet = transform.inverse_and_log_det(event)
    np.testing.assert_allclose(restored, observed, rtol=0, atol=1e-12)
    assert np.ptp(np.asarray(event[2, 1])) > 2 * np.pi
    assert np.max(np.abs(np.diff(event[2, 1]))) < np.pi
    np.testing.assert_allclose(forward_logdet, expected, atol=1e-12)
    np.testing.assert_allclose(inverse_logdet, -expected, atol=1e-12)
    assert not transform.is_constant_jacobian
    assert not transform.is_constant_log_det
    assert jax.config.jax_enable_x64


def test_reference_discrepancy_transfers_to_reflection_fit_and_qoi():
    result = example.run(n_frequency=150, n_transfer=30)
    assert abs(result['fitted_load'] - 76.0) < 1.0
    assert result['relative_std_error'] < 0.05
    assert np.isfinite(result['P_q']).all()
    assert np.isfinite(result['G_q']).all()
    assert result['delta_std'] > 0
    additive = np.asarray(result['additive_residual'].real)
    logarithmic = np.asarray(result['log_residual'][0])
    crossings = lambda values: np.count_nonzero(np.diff(np.signbit(values)))
    assert crossings(additive) > 20
    assert crossings(logarithmic) < crossings(additive) / 4
