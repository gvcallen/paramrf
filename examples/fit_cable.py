import skrf as rf

from pmrf.statistics_legacy.parameters import ParameterSet
from pmrf.models.lines import DatasheetCoaxial
from pmrf.parameter import norm, uniform, fixed
from pmrf.fitting import ScipyFitter

# Create the model
coax = DatasheetCoaxial(name='coax')

# Setup the parameters
parameters = {
    'length': norm(10.0, 0.5),
    'zn': norm(49.0, 51.0),
    'epr': fixed(1.0),
    'epr_slope': fixed(0.0),
    'k1': uniform(0.0, 2.0),
    'k2': fixed(0.0),
}

# Initialize the fitter
fitter = ScipyFitter(
    model=coax,
    measured=rf.Network('examples/data/10m_cable.s2p', f_unit='MHz'),
    params=parameters
)

# Run the fit
fitter.fit()

# a, w = 0.8, 0.4
# params = ParameterSet(
#     columns=['name', 'dist'],
#     data=[
#         ['coax_din', uniform(1.12e-3*a, 1.12e-3*w)],
#         ['coax_dout', uniform(3.2e-3*a, 3.2e-3*w)],
#         ['coax_len', uniform(10*a, 10*w)],
#         ['coax_epr', uniform(1.45*a, 1.45*w)],
#         ['coax_tand', uniform(0.0, 0.1)],
#         ['coax_rho', uniform(1.6e-8*a, 1.6e-8*w)],
#     ]
# )

# fitter = BaseFitter(
#     output_path='output_cable',
#     param_set=params,
#     model=[coax],
#     measured=[rf.Network('examples/data/10m_cable.s2p', f_unit='MHz')],
#     use_measured_frequency=True,
#     ports=[(0, 0), (0, 1), (1, 0), (1, 1)], # We could specify which ports we want to fit on specifically
#     solver='Nelder-Mead',
#     max_iterations=10000,
# )

# fitter.plotter.plot_S('s_initial')
# fitter.fit_params()
# fitter.plotter.plot_S('s_opt')
