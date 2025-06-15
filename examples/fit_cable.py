import skrf as rf
from scipy.stats import uniform

from pmrf.statistics.parameters import ParameterSet
from pmrf.models.lines import PhysicalCoaxial
from pmrf.fit import ModelFitter

# This example shows how to setup a model "manually" and fit that model to data. Here we fit the "PhysicalCoaxial" model to a 10m cable lab measurement.
coax = PhysicalCoaxial(name='coax')

# Setup parameter priors/bounds. We could also leave this out and let the NetworkFitter generate the bounds.
# Parameters follow the model name, then the infix '_' in the fitter settings, and then the parameters of the Model (PhysicalCoaxialLine).
# Note that for non-uniform distribution, frequentist solvers use the 1% and 99% bounds of the specified distribution as their min/max.
a, w = 0.8, 0.4
params = ParameterSet(
    columns=['name', 'dist'],
    data=[
        ['coax_din', uniform(1.12e-3*a, 1.12e-3*w)],
        ['coax_dout', uniform(3.2e-3*a, 3.2e-3*w)],
        ['coax_len', uniform(10*a, 10*w)],
        ['coax_epr', uniform(1.45*a, 1.45*w)],
        ['coax_tand', uniform(0.0, 0.1)],
        ['coax_rho', uniform(1.6e-8*a, 1.6e-8*w)],
    ]
)

fitter = ModelFitter(
    output_path='output_cable',
    param_set=params,
    model=[coax],
    measured=[rf.Network('examples/data/10m_cable.s2p', f_unit='MHz')],
    use_measured_frequency=True,
    ports=[(0, 0), (0, 1), (1, 0), (1, 1)], # We could specify which ports we want to fit on specifically
    solver='Nelder-Mead',
    max_iterations=10000,
)

fitter.plotter.plot_S('s_initial')
fitter.fit_params()
fitter.plotter.plot_S('s_opt')
