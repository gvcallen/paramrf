import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.models import Circuit, GridPortDiscrepancy, Load, Port, PortCorrected, RLGCLine, SModel


FREQUENCY = prf.Frequency(1, 2, 5, unit='ghz')


def test_zero_discrepancy_reproduces_line_exactly():
    line = RLGCLine(length=0.2, R=1.0, name='cable')
    discrepancy = GridPortDiscrepancy.zeros(FREQUENCY, ('11', '22', 's21'))
    corrected = PortCorrected(line, discrepancy)
    np.testing.assert_array_equal(corrected.s(FREQUENCY), line.s(FREQUENCY))
    assert corrected.number_of_ports == 2


@pytest.mark.parametrize('value, factor', [(np.log(1.01), 1.01), (0.1j, np.exp(0.1j))])
def test_symmetric_transmission_changes_both_directions(value, factor):
    line = RLGCLine(length=0.2, R=1.0)
    values = np.zeros((1, 2, FREQUENCY.npoints))
    values[0, 0] = value.real
    values[0, 1] = value.imag
    discrepancy = GridPortDiscrepancy(values, FREQUENCY, ('s21',), 2)
    corrected = PortCorrected(line, discrepancy).s(FREQUENCY)
    original = line.s(FREQUENCY)
    for i, j in [(0, 1), (1, 0)]:
        np.testing.assert_allclose(corrected[:, i, j] / original[:, i, j], factor, rtol=0, atol=1e-12)
    # RLGC conversion has independent floating-point paths for S12 and S21;
    # exact equality of the correction is checked on an exactly reciprocal S.
    np.testing.assert_array_equal(discrepancy(FREQUENCY)[:, 0, 1], discrepancy(FREQUENCY)[:, 1, 0])
    np.testing.assert_array_equal(corrected[:, 0, 0], original[:, 0, 0])


def test_reflection_and_antisymmetric_blocks_leave_uncovered_ports_zero():
    discrepancy = GridPortDiscrepancy(
        np.array([[[0.2] * 5, [0.3] * 5], [[0.4] * 5, [-0.1] * 5]]),
        FREQUENCY, ('11', 'a21'), 3,
    )
    delta = discrepancy(FREQUENCY)
    np.testing.assert_allclose(delta[:, 0, 0], 0.2 + 0.3j, atol=1e-12)
    np.testing.assert_allclose(delta[:, 1, 0] - delta[:, 0, 1], 0.8 - 0.2j, atol=1e-12)
    np.testing.assert_array_equal(delta[:, 2, :], 0)
    np.testing.assert_array_equal(delta[:, :, 2], 0)


def test_reference_impedance_correction_then_renormalization():
    import skrf

    line = RLGCLine(length=0.2, R=1.0)
    discrepancy = GridPortDiscrepancy(
        np.array([[[0.02] * 5, [0.03] * 5], [[np.log(1.01)] * 5, [0.1] * 5]]),
        FREQUENCY, ('11', 's21'), 2,
    )
    # Independently construct the specified correction and let scikit-rf
    # renormalise power waves at real reference impedances.
    expected = np.array(line.s(FREQUENCY, z0=60))
    expected[:, 0, 0] += 0.02 + 0.03j
    expected[:, 0, 1] *= 1.01 * np.exp(0.1j)
    expected[:, 1, 0] *= 1.01 * np.exp(0.1j)
    network = skrf.Network(f=np.asarray(FREQUENCY.f), s=expected, z0=60)
    network.renormalize(75)
    np.testing.assert_allclose(PortCorrected(line, discrepancy, z0=60).s(FREQUENCY, z0=75), network.s, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('grid', [prf.Frequency(1, 3, 5, 'GHz'), prf.Frequency(1, 2, 3, 'GHz')])
def test_grid_mismatch_raises(compiled, grid):
    discrepancy = GridPortDiscrepancy.zeros(FREQUENCY, ('s21',))
    evaluate = jax.jit(lambda f: discrepancy(f)) if compiled else discrepancy
    with pytest.raises(Exception, match='own frequency grid'):
        evaluate(grid).block_until_ready()


def test_exact_symmetric_transmission_ratio():
    model = SModel(np.broadcast_to([[0.1, 0.8j], [0.8j, 0.2]], (5, 2, 2)), FREQUENCY, 50)
    discrepancy = GridPortDiscrepancy(np.full((1, 2, 5), 0.1), FREQUENCY, ('s21',), 2)
    original = model.s(FREQUENCY)
    corrected = PortCorrected(model, discrepancy).s(FREQUENCY)
    np.testing.assert_array_equal(corrected[:, 0, 1] / original[:, 0, 1], corrected[:, 1, 0] / original[:, 1, 0])


def test_empty_blocks_are_identity_and_dense_constructor_infers_ports():
    empty = GridPortDiscrepancy.zeros(FREQUENCY, (), number_of_ports=2)
    np.testing.assert_array_equal(empty(FREQUENCY), np.zeros((5, 2, 2)))
    dense = GridPortDiscrepancy(np.zeros((1, 2, 5)), FREQUENCY, ('s21',))
    assert dense.number_of_ports == 2


@pytest.mark.parametrize('blocks', [('12',), ('s12',), ('s11',), ('s21', 's21')])
def test_invalid_blocks_raise(blocks):
    with pytest.raises(ValueError, match='block|transmission'):
        GridPortDiscrepancy.zeros(FREQUENCY, blocks, number_of_ports=2)


@pytest.mark.parametrize('values', [np.zeros((1, 5, 2)), np.zeros((1, 2, 5), dtype=complex)])
def test_values_require_real_event_layout(values):
    with pytest.raises(ValueError, match='real with shape'):
        GridPortDiscrepancy(values, FREQUENCY, ('s21',))


def _circuit(component):
    p1, p2 = Port(), Port()
    return Circuit([[(p1, 0), (component, 0)], [(component, 1), (p2, 0)]])


def test_zero_discrepancy_reproduces_subcircuit_exactly():
    circuit = _circuit(RLGCLine(length=0.2, R=1.0))
    corrected = PortCorrected(circuit, GridPortDiscrepancy.zeros(FREQUENCY, ('11', '22', 's21')))
    np.testing.assert_array_equal(corrected.s(FREQUENCY), circuit.s(FREQUENCY))


def test_parameter_names_and_circuit_keep_base_parameters():
    line = RLGCLine(length=0.2, R=1.0, name='cable')
    corrected = PortCorrected(line, GridPortDiscrepancy.zeros(FREQUENCY, ('s21',)))
    assert set(prf.params(corrected)) == set(prf.params(line)) | {'discrepancy.values'}
    circuit = _circuit(corrected)
    original = _circuit(line)
    assert set(prf.params(circuit)) == set(prf.params(original)) | {'cable.discrepancy.values'}
    assert 'cable.R' in prf.params(circuit)
    updated = prf.update(circuit, 'cable.discrepancy.values', value=jnp.full((1, 2, 5), 0.05))
    assert not np.allclose(updated.s(FREQUENCY), circuit.s(FREQUENCY))


def test_terminated_cascade_jit_and_derivative():
    corrected = PortCorrected(
        RLGCLine(length=0.2, R=1.0, name='cable'),
        GridPortDiscrepancy.zeros(FREQUENCY, ('11', 's21')),
    )
    circuit = (corrected ** RLGCLine(length=0.1, R=2.0, name='second')).terminated(Load(gamma=3.0/13.0))
    np.testing.assert_allclose(jax.jit(lambda: circuit.s(FREQUENCY))(), circuit.s(FREQUENCY), rtol=1e-12, atol=1e-12)
    evaluate = lambda m: jnp.sum(jnp.abs(m.s(FREQUENCY)) ** 2)
    (gradient,) = prf.derivative(evaluate, circuit)
    derivatives = prf.values(gradient)
    key = next(key for key in derivatives if key.endswith('discrepancy.values'))
    step = np.zeros((2, 2, 5))
    step[1, 0, 2] = 1e-5
    plus = evaluate(prf.update(circuit, key, value=step))
    minus = evaluate(prf.update(circuit, key, value=-step))
    np.testing.assert_allclose(derivatives[key][1, 0, 2], (plus - minus) / 2e-5, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(jax.grad(lambda x: evaluate(prf.update(circuit, key, value=x)))(jnp.zeros((2, 2, 5))), derivatives[key], rtol=1e-12, atol=1e-12)
