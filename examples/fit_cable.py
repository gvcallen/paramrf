import skrf as rf

from pmrf.models.lines import PhysicalCoaxial
from pmrf.parameter import norm, uniform, fixed
from pmrf.fitting import ScipyFitter, OptaxFitter
from pmrf.fitting import Feature

# Create the model
model = PhysicalCoaxial()

wa, wb = 0.8, 1.2
params = {
    'din': uniform(1.12e-3*wa, 1.12e-3*wb),
    'dout': uniform(3.2e-3*wa, 3.2e-3*wb),
    'length': uniform(10*wa, 10*wb),
    'epr': uniform(1.45*wa, 1.45*wb),
    'tand': uniform(0.0, 0.1),
    'rho': uniform(1.6e-8*wa, 1.6e-8*wb),
}

measured = rf.Network('examples/data/10m_cable.s2p', f_unit='MHz')

# features = [
#     Feature(mode='complex', ports=(0, 0)), Feature(mode='magnitude', ports=(0, 0)),
#     Feature(mode='complex', ports=(0, 1)), Feature(mode='magnitude', ports=(0, 1)),
#     Feature(mode='complex', ports=(1, 0)), Feature(mode='magnitude', ports=(1, 0)),
#     Feature(mode='complex', ports=(1, 1)), Feature(mode='magnitude', ports=(1, 1))
# ]

features = ['s11', 's11_mag', 's12', 's12_mag', 's21', 's21_mag', 's22', 's22_mag']

fitter = ScipyFitter(
    model=model,
    measured=measured,
    params=params,
    features=features,
)

fitter.run()