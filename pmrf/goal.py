from typing import Callable

import equinox as eqx
import jax.numpy as jnp

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.metrics import metric_from_alias
from pmrf.constants import FeatureSpec
from pmrf.extractor import Extractor, make_extractors, extract_multiple_features

class Goal(eqx.Module):
    """
    A goal function, used for comparing a model's features against a target.
    
    This class extracts specific features from a Model, compares them against a 
    defined target using a logical operator, computes the error using a metric 
    function, applies an optional mask, aggregates the result, and scales it by a weight.
    """
    # Extracting and clipping against the target
    extractors: list[Extractor] = eqx.field(static=True)
    operator: str = eqx.field(static=True)
    target: jnp.ndarray

    # Error computation and transformation
    metric_fn: Callable = eqx.field(static=True)
    mask: jnp.ndarray | None = eqx.field(static=True)
    weight: float = eqx.field(static=True)

    def __init__(
        self,
        features: FeatureSpec | list[FeatureSpec] | list[Extractor],
        operator: str = '==',
        target: float | jnp.ndarray = 0.0,
        *,
        metric_fn: Callable | str = 'rms',
        mask: jnp.ndarray | None = None,
        weight: float = 1.0,
    ):
        extractors = features
        is_extractor_list = isinstance(extractors, list) and len(extractors) > 0 and isinstance(extractors[0], Extractor)
        if not is_extractor_list:
            extractors = make_extractors(extractors)
            
        if operator not in ('<', '>', '=='):
            raise ValueError(f"Operator must be '<', '>', or '=='. Got '{operator}'")
            
        if isinstance(target, (float, int)):
            target = jnp.array(target)
        if isinstance(metric_fn, str):
            metric_fn = metric_from_alias(metric_fn)
        
        self.extractors = extractors
        self.operator = operator
        self.target = target
        self.metric_fn = metric_fn
        self.mask = mask
        self.weight = weight

    def __call__(self, model: Model, frequency: Frequency) -> jnp.ndarray:
        """
        Evaluates the model against the goal at the specified frequency.

        Parameters
        ----------
        model : Model
            The ParamRF model to evaluate.
        frequency : Frequency
            The frequency array/object to evaluate the features against.

        Returns
        -------
        jnp.ndarray
            The computed, aggregated scalar cost for this goal.
        """
        # Extract the features
        if len(self.extractors) == 1:
            pred = self.extractors[0](model, frequency)
        else:
            pred = extract_multiple_features(self.extractors, model, frequency)

        # Apply operator logic (differentiable thresholding)
        if self.operator == '==':
            effective_pred = pred
        elif self.operator == '<':
            effective_pred = jnp.maximum(pred, self.target)
        elif self.operator == '>':
            effective_pred = jnp.minimum(pred, self.target)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

        # Compute the element-wise error metric
        error = self.metric_fn(self.target, effective_pred)

        # Apply the mask, aggregation, and weight
        if self.mask is not None:
            error = error * self.mask

        error = jnp.sum(error)
        return error * self.weight