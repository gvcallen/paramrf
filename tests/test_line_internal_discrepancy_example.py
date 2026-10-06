"""Execute each public recipe in the internal discrepancy example."""

import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    'line_internal_discrepancy_example',
    Path(__file__).parents[1] / 'docs/examples/line_internal_discrepancy.py',
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_comparison_internal_correction_wins_on_s21_and_eta():
    result = example.compare_transfer()
    internal = result['internal']
    assert internal['rms_s21'] < result['uncorrected']['rms_s21']
    assert internal['rms_s21'] < result['port']['rms_s21']
    assert abs(internal['eta_bias']) < abs(result['uncorrected']['eta_bias'])
    assert abs(internal['eta_bias']) < abs(result['port']['eta_bias'])


def test_reference_route_improves_transfer_s21():
    result = example.reference_route()
    assert np.isfinite(result['log_prior'])
    assert result['rms_s21'] < example.compare_transfer()['uncorrected']['rms_s21']


def test_s_only_route_reports_coverage_and_improves_transfer_s21():
    result = example.s_only_route()
    assert 0 <= result['coverage'] <= 1
    assert result['mean'].shape == result['sigma'].shape == result['injected'].shape == (2, 2, 7)
    assert result['coverage'] == np.mean(
        np.abs(result['injected'] - result['mean']) <= 2 * result['sigma'])
    assert result['rms_s21'] < example.compare_transfer()['uncorrected']['rms_s21']
