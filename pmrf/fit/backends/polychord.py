import jax.numpy as jnp
from typing import Any
from pmrf.models.model import Model
from pmrf.fitting.bayesian import BayesianFitter
from pmrf.backends.polychord import run_polychord, PolyChordIOMixin

class PolyChordFitter(BayesianFitter, PolyChordIOMixin):
    """
    Bayesian fitter using the PolyChord nested sampling algorithm.
    
    This backend uses ``pypolychord`` to perform slice sampling, calculating 
    both the global evidence (logZ) and generating samples from the posterior 
    distribution.
    """
    def execute(
        self, 
        target: jnp.ndarray, 
        **kwargs
    ) -> tuple[Model, Any]:
        
        param_names = self.model.flat_param_names() + list(self.likelihood_params.keys())
        
        bound_ll_fn = lambda x: self.log_likelihood(x, target)
        
        return run_polychord(
            model=self.model,
            log_likelihood_fn=bound_ll_fn,
            icdf_fn=self.icdf,
            param_names=param_names,
            logger=self.logger,
            output_path=self.output_path,
            output_root=self.output_root or 'polychord',
            **kwargs
        )