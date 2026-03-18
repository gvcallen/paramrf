"""
Test that fitting works.
"""

from pathlib import Path

import skrf as rf
import numpy as np

import pmrf as prf
from pmrf.core import CoaxialLine
from pmrf.parameters import Uniform, RelativeNormal, Fixed
from pmrf.fitting import SciPyMinimizeFitter

TEST_DIR = Path(__file__).parent

def test_fitting():
    # Load the measured data and setup the model
    data_path = TEST_DIR / "data" / "10m_cable.s2p"
    measured = rf.Network(data_path, f_unit='MHz')
    model = CoaxialLine(
        din = RelativeNormal(1.12, 0.05, scale=1e-3),
        dout = RelativeNormal(3.2, 0.05, scale=1e-3),
        epr = Fixed(1.384),
        rho = RelativeNormal(1.6, 0.05, scale=1e-8),
        tand = Uniform(0.0, 0.01, value=0.0, scale=0.01),
        length = RelativeNormal(10.0, 0.05),
        mur = Fixed(1.0),
    )

    # Initialize the fitter
    fitter = SciPyMinimizeFitter(model)

    # Run the fit
    fitted_model, fit_results = fitter.run(measured)

    # Assert the residuals are less than -30 dB
    residuals = measured.s - fitted_model.s(prf.Frequency.from_skrf(measured.frequency))
    max_residual_db = np.max(20*np.log10(np.abs(residuals)))
    assert max_residual_db < -30