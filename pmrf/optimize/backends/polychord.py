from typing import Any
from pmrf.models.model import Model
from pmrf.optimize.bayesian import BayesianOptimizer
from pmrf.backends.polychord import run_polychord, PolyChordIOMixin

class PolyChordOptimizer(BayesianOptimizer, PolyChordIOMixin):
    """
    Bayesian optimizer using the PolyChord nested sampling algorithm.
    
    This backend maps the goal-oriented feasible design space by using 
    slice sampling. It finds the volume of the parameter space that satisfies 
    the optimization goals.
    """
    def execute(
        self, 
        **kwargs
    ) -> tuple[Model, Any]:
        
        param_names = self.model.flat_param_names()
        
        bound_ll_fn = lambda x: self.log_likelihood(x)
        
        return run_polychord(
            model=self.model,
            log_likelihood_fn=bound_ll_fn,
            icdf_fn=self.icdf,
            param_names=param_names,
            logger=self.logger,
            output_path=self.output_path,
            output_root=self.output_root or 'polychord_opt',
            **kwargs
        )