import skrf as rf

from pmrf.statistics import UniformPDF, ParameterSet
from pmrf.modeling.models import PhysicalCoax
from pmrf.fitting import CircuitFitter

# This example shows how to setup a model "manually" and fit that model to data. Here we fit the "PhysicalCoax" model to the 10m cable lab measurement.
# Note that we are also able to define our own models using ParametricNetwork -derived class.
coax = PhysicalCoax(name='coax')

# Setup parameter priors/bounds. We could also leave this out and let the CircuitFitter generate the bounds.
# Parameters follow the model name, then the infix '_' in the fitter settings, and then the parameters of the ParametricNetwork (DatasheetCoax).
# Note that priors aren't actually used as such for frequentist solvers (only ther min/max is used).
wa, wb = 0.8, 1.2
params = ParameterSet(
    columns=['name', 'pdf'],
    data=[
        ['coax_din', UniformPDF(1.12e-3*wa, 1.12e-3*wb)],
        ['coax_dout', UniformPDF(3.2e-3*wa, 3.2e-3*wb)],
        ['coax_len', UniformPDF(10*wa, 10*wb)],
        ['coax_epr', UniformPDF(1.45*wa, 1.45*wb)],
        ['coax_tand', UniformPDF(0.0, 0.1)],
        ['coax_rho', UniformPDF(1.6e-8*wa, 1.6e-8*wb)],
    ]
)

fitter = CircuitFitter(
    output_path='output_cable',
    param_set=params,
    models=[coax],
    measured=[rf.Network('examples/data/10m_cable.s2p', f_unit='MHz')],
    use_measured_frequency=True,
    ports=[(0, 0), (0, 1), (1, 0), (1, 1)], # We could specify which ports we want to fit on specifically
    solver='Nelder-Mead',
    max_iterations=10000,
)

fitter.plotter.plot_S('s_initial')
fitter.fit_params()
fitter.plotter.plot_S('s_opt')
