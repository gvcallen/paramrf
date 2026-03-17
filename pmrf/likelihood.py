from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.extractor import Extractor, make_extractors, extract_multiple_features
from pmrf.parameters import Parameter
from pmrf.constants import FeatureSpec

      
class Likelihood(eqx.Module):
    extractors: list[Extractor] = eqx.field(static=True)
    observed: jnp.ndarray

    dist_fn: Callable[[jnp.ndarray], dist.Distribution] = eqx.field(static=True)
    params: dict[str, Parameter]

    # Error computation and transformation
    metric_fn: Callable = eqx.field(static=True)
    mask: jnp.ndarray | None = eqx.field(static=True)
    aggregate_fn: Callable = eqx.field(static=True)
    transform_fn: Callable | None = eqx.field(static=True)
    weight: float = eqx.field(static=True)    

    def __init__(
        self, 
        features: FeatureSpec | list[FeatureSpec] | list[Extractor],
        observed: jnp.ndarray,
        *,
        dist_fn: Callable | str = dist.Normal,
        params: dict[str, Parameter],
    ):
        extractors = features
        is_extractor_list = isinstance(extractors, list) and len(extractors) > 0 and isinstance(extractors[0], Extractor)
        if not is_extractor_list:
            extractors = make_extractors(extractors)

        self.extractors = extractors
        self.observed = jnp.atleast_1d(jnp.array(observed))
        self.dist_fn = dist_fn
        self.params = params

    def __call__(self, model: Model, frequency: Frequency) -> jnp.ndarray:
        if len(self.extractors) == 1:
            pred = self.extractors[0](model, frequency)
        else:
            pred = extract_multiple_features(self.extractors, model, frequency)
            
        return self.dist_fn(pred, **self.params).log_prob(self.observed).sum()