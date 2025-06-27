import skrf as rf

from models.lines import PhysicalCoaxial
from pmrf.parameters import Normal, Uniform, Fixed
from pmrf.fitting import ScipyMinimizeFitter, OptaxFitter
from pmrf.fitting import Feature

# Load the measured data
measured = rf.Network('examples/data/10m_cable.s2p', f_unit='MHz')

# Define parameters/bounds and initialize the model
wa, wb = 0.8, 1.2
params = {
    'din': Uniform(1.12e-3*wa, 1.12e-3*wb),
    'dout': Uniform(3.2e-3*wa, 3.2e-3*wb),
    'length': Uniform(10*wa, 10*wb),
    'epr': Uniform(1.45*wa, 1.45*wb),
    'tand': Uniform(0.0, 0.1),
    'rho': Uniform(1.6e-8*wa, 1.6e-8*wb),
}
model = PhysicalCoaxial().with_params(params, all_check=True)


# features = [
#     Feature(mode='complex', ports=(0, 0)), Feature(mode='magnitude', ports=(0, 0)),
#     Feature(mode='complex', ports=(0, 1)), Feature(mode='magnitude', ports=(0, 1)),
#     Feature(mode='complex', ports=(1, 0)), Feature(mode='magnitude', ports=(1, 0)),
#     Feature(mode='complex', ports=(1, 1)), Feature(mode='magnitude', ports=(1, 1))
# ]

features = ['s11', 's11_mag', 's12', 's12_mag', 's21', 's21_mag', 's22', 's22_mag']

fitter = ScipyMinimizeFitter(
    model=model,
    measured=measured,
    params=params,
    features=features,
)

fitter.run()