from pmrf.models.model import Model
from pmrf.optimize.frequentist import FrequentistOptimizer
from pmrf.backends.optax import run_optax

class OptaxOptimizer(FrequentistOptimizer):
    """
    Frequentist optimizer using the Optax and JAX backend.
    
    This class leverages JAX's automatic differentiation and Optax's gradient 
    transformations to optimize the model parameters. Box constraints are 
    enforced via projected gradient descent (clipping parameters after each step).
    """    
    def execute(
        self, 
        **kwargs
    ) -> tuple[Model, dict]:
        return run_optax(
            model=self.model,
            cost_fn=self.cost,
            logger=self.logger,
            **kwargs
        )