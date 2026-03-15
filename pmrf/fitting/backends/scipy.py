import jax.numpy as jnp
from scipy.optimize import OptimizeResult

from pmrf.fitting.frequentist import FrequentistFitter
from pmrf.models.model import Model
from pmrf.backends.scipy import run_scipy_minimize

class SciPyMinimizeFitter(FrequentistFitter):
    """
    Frequentist fitter using the SciPy minimize backend with JAX acceleration.
    
    This class wraps ``scipy.optimize.minimize``, executing the optimization
    in a normalized parameter space [0, 1] for fully bounded parameters, 
    while preserving natural scaling for unbounded or semi-bounded parameters.
    """
    def execute(
        self, 
        target: jnp.ndarray, 
        **kwargs
    ) -> tuple[Model, OptimizeResult]:
                
        # Bind the target to the cost function
        bound_cost_fn = lambda x: self.cost(x, target)

        return run_scipy_minimize(
            model=self.model,
            cost_fn=bound_cost_fn,
            logger=self.logger,
            **kwargs
        )