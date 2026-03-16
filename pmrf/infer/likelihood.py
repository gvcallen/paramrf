import equinox as eqx
import jax.numpy as jnp
from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.optimize.goal import Goal
from pmrf.features import Extractor, make_extractors, extract_multiple_features
from pmrf.parameters import Parameter
from pmrf.constants import FeatureSpec
   
class CombinedLikelihood(eqx.Module):
    """Safely evaluates multiple log-likelihood terms and sums them."""
    terms: tuple

    def __call__(self, model: Model, frequency: Frequency) -> jnp.ndarray:
        return jnp.sum(jnp.array([term(model, frequency) for term in self.terms]))    
    
class GaussianLikelihood(eqx.Module):
    """
    Evaluates the log-likelihood of a Model against measured data.
    """
    extractors: list[Extractor] = eqx.field(static=True)
    measured_data: jnp.ndarray
    
    # Statistical noise parameter (will be automatically flattened by JAX!)
    sigma: Parameter 

    def __init__(
        self, 
        features: FeatureSpec | list[FeatureSpec] | list[Extractor], 
        measured_data: jnp.ndarray,
        sigma: Parameter
    ):
        extractors = features
        is_extractor_list = isinstance(extractors, list) and len(extractors) > 0 and isinstance(extractors[0], Extractor)
        if not is_extractor_list:
            extractors = make_extractors(extractors)
            
        self.extractors = extractors
        self.measured_data = jnp.atleast_1d(jnp.array(measured_data))
        self.sigma = sigma

    def __call__(self, model: Model, frequency: Frequency) -> jnp.ndarray:
        # 1. Extract the predicted features from the RF model
        if len(self.extractors) == 1:
            pred = self.extractors[0](model, frequency)
        else:
            pred = extract_multiple_features(self.extractors, model, frequency)
            
        # 2. Evaluate the probability density
        return dist.Normal(loc=pred, scale=self.sigma.value).log_prob(self.measured_data).sum()