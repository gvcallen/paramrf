import logging
import skrf as rf
import matplotlib.pyplot as plt

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

from pmrf.models.lines import CoaxialLine
from pmrf.parameters import Uniform, Fixed, PercentNormal
from pmrf.fitting import PolyChordFitter

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

# Initialize the fitter. Note that the features need to align with the likelihood
fitter = PolyChordFitter(
    model=model,
    output_path='output_fit_cable_polychord',
)

# Run the fit and plot the results. We use 1x the number of parameters for the live fit.
# Note that you can should run this script using mpi if possible e.g. `mpirun -np 6 python fit_cable_polychord.py` for 6 processes.
results = fitter.run(measured, nlive_factor=1, save_results=True)
if rank == 0:
    results.fitted_model.to_skrf(measured.frequency).plot_s_db(m=0, n=0)
    measured.plot_s_db(m=0, n=0)
    plt.show()