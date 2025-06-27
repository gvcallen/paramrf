import logging
import skrf as rf
import matplotlib.pyplot as plt

from pmrf.models.lines import PhysicalCoaxial
from pmrf.parameters import Uniform, Fixed
from pmrf.fitting import PolychordFitter

logging.basicConfig(level=logging.INFO)

# Load the measured data
measured = rf.Network('paramrf/examples/data/10m_cable.s2p', f_unit='MHz')

# Define parameters/bounds and initialize the model
wa, wb = 0.8, 1.2
params = {
    'din': Uniform(1.12e-3*wa, 1.12e-3*wb),
    'dout': Uniform(3.2e-3*wa, 3.2e-3*wb),
    'length': Uniform(9.0, 11.0),
    'epr': Uniform(1.35, 1.45),
    'tand': Uniform(0.0, 0.1),
    'rho': Uniform(1.0e-9, 1.6e-7),
    'mur': Fixed(1.0),
}
model = PhysicalCoaxial().with_params(params, all_check=True)

# Initialize the fitter. Note that the features need to align with the likelihood
fitter = PolychordFitter(
    model=model,
    measured=measured,
    features=['s11_re', 's11_im'],    
)

# Run the fit and plot the results. We use 10x the number of parameters for the live fit
result = fitter.run(nlive=60)
result.model.to_skrf(measured.frequency).plot_s_db(m=0, n=0)
measured.plot_s_db(m=0, n=0)
plt.show()