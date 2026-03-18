import jax.numpy as jnp
from pmrf.core import Model
from pmrf.fitting.frequentist import FrequentistFitter
from pmrf.backends.optax import run_optax

class OptaxFitter(FrequentistFitter):
    """
    Frequentist fitter using the Optax and JAX backend.
    
    This class leverages JAX's automatic differentiation and Optax's gradient 
    transformations to optimize the model parameters. Box constraints are 
    enforced via projected gradient descent (clipping parameters after each step).
    """    
    def execute(
        self, 
        target: jnp.ndarray,
        **kwargs
    ) -> tuple[Model, dict]:
        
        # Bind the target to the cost function
        bound_cost_fn = lambda x: self.cost(x, target)

        return run_optax(
            model=self.model,
            cost_fn=bound_cost_fn,
            logger=self.logger,
            **kwargs
        )