import matplotlib.pyplot as plt
import logging
import skrf as rf
import jax.numpy as jnp

import pmrf as prf
from pmrf.models import CoaxialLine
from pmrf.parameters import Uniform, Fixed, PercentNormal
from pmrf.fitting import SciPyMinimizeFitter

logging.basicConfig(level=logging.INFO)

# Load the measured data and setup the model
measured = rf.Network('data/10m_cable.s2p', f_unit='MHz')
model = CoaxialLine(
    din = PercentNormal(1.12, 5.0, scale=1e-3),
    dout = PercentNormal(3.2, 5.0, scale=1e-3),
    epr = PercentNormal(1.45, 5.0, n=2),
    rho = PercentNormal(1.6, 5.0, scale=1e-8),
    tand = Uniform(0.0, 0.01, value=0.0, scale=0.01, n=2),
    mur = Fixed(1.0),
    length = PercentNormal(10.0, 5.0),
    epr_model='bpoly',
    tand_model='bpoly',
)

# Initialize the fitter. We fit on the real and imaginary and combine their results
fitter = SciPyMinimizeFitter(
    model=model,
    features=['s11_re', 's11_im'],
    error_fn=[prf.l2_norm_ax0, jnp.sum, prf.mag_2_db],
)

# Run the fit
fitted_model, fit_results = fitter.run(measured, optimizer='Nelder-Mead')
model_ntwk = fitted_model.to_skrf(measured.frequency)

# Plot some results
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
model_ntwk.plot_s_db(m=0, n=0, ax=axes[0])
measured.plot_s_db(m=0, n=0, ax=axes[0])
model_ntwk.plot_s_deg(m=0, n=0, ax=axes[1])
measured.plot_s_deg(m=0, n=0, ax=axes[1])
model_ntwk.plot_s_re(m=0, n=0, ax=axes[2])
measured.plot_s_re(m=0, n=0, ax=axes[2])
model_ntwk.plot_s_im(m=0, n=0, ax=axes[3])
measured.plot_s_im(m=0, n=0, ax=axes[3])

plt.show()