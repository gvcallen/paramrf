import os
import logging
import skrf as rf
import matplotlib.pyplot as plt

from pmrf.models.lines import PhysicalCoaxial
from pmrf.parameters import Uniform, Fixed, PercentNormal
from pmrf.fitting import BlackjaxNSFitter

logging.basicConfig(level=logging.INFO)

# Load the measured data and setup the model
measured = rf.Network('data/10m_cable.s2p', f_unit='MHz')
model = PhysicalCoaxial(
    din = PercentNormal(1.12, 5.0, scale=1e-3),
    dout = PercentNormal(3.2, 5.0, scale=1e-3),
    epr = PercentNormal(1.45, 5.0, n=2),
    rho = PercentNormal(1.6, 5.0, scale=1e-8),
    tand = Uniform(0.0, 0.01, value=0.0, scale=0.01, n=2),
    mur = Fixed(1.0),
    length = PercentNormal(10.0, 5.0),
    epr_model='bpoly',
    epr_order=2,
    tand_model='bpoly',
    tand_order=2
)

# Initialize the fitter. Note that the features need to align with the likelihood
fitter = BlackjaxNSFitter(
    model=model,
    measured=measured,
    features=['s11_re', 's11_im'],
)

# Run the fit and plot the results. We use 10x the number of parameters for the live fit
result = fitter.run()
os.makedirs('output/fit_cable_blackjax', exist_ok=True)
result.to_hdf5('output/fit_cable_blackjax/cable_fit.hdf5')


# result.model.to_skrf(measured.frequency).plot_s_db(m=0, n=0)
# measured.plot_s_db(m=0, n=0)
# plt.show()