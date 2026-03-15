from pmrf.models.model import Model
from pmrf.optimize.frequentist import FrequentistOptimizer
from scipy.optimize import OptimizeResult

from pmrf.backends import run_scipy_minimize

class SciPyMinimizeOptimizer(FrequentistOptimizer):
    """
    Frequentist optimizer using the SciPy minimize backend with JAX acceleration.
    
    This class wraps ``scipy.optimize.minimize``, executing the optimization
    in a normalized parameter space [0, 1] for fully bounded parameters, 
    while preserving natural scaling for unbounded or semi-bounded parameters.
    """    
    def execute(
        self, 
        **kwargs
    ) -> tuple[Model, OptimizeResult]:
        return run_scipy_minimize(
            **kwargs
        )