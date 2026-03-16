from typing import Callable, Sequence

import jsonpickle
import equinox as eqx
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.metrics import metric_from_alias
from pmrf.aggregation import aggregate_from_alias
from pmrf.constants import FeatureSpec
from pmrf.features import Extractor, make_extractors, extract_multiple_features

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
    aggregate_fn: Callable = eqx.field(static=True)
    transform_fn: Callable | None = eqx.field(static=True)
    weight: float = eqx.field(static=True)

    def __init__(
        self,
        features: FeatureSpec | list[FeatureSpec] | list[Extractor],
        operator: str = '==',
        target: float | jnp.ndarray = 0.0,
        *,
        metric_fn: Callable | str = 'rms',
        aggregate_fn: Callable | str = 'rms',
        transform_fn: Callable | None = 'db',
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
        if isinstance(aggregate_fn, str):
            aggregate_fn = aggregate_from_alias(aggregate_fn)
        if transform_fn is not None and isinstance(transform_fn, str):
            if transform_fn == 'db':
                transform_fn = lambda x: 20*jnp.log10(x)
            elif transform_fn == 'linear':
                transform_fn = None
        
        self.extractors = extractors
        self.operator = operator
        self.target = target
        self.metric_fn = metric_fn
        self.aggregate_fn = aggregate_fn
        self.transform_fn = transform_fn
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
        cost = self.aggregate_fn(error)

        if self.transform_fn is not None:
            cost = self.transform_fn(cost)
        
        return cost * self.weight
        
    def to_json(self) -> str:
        """Serialize the goal to a JSON string for HDF5 storage."""
        return jsonpickle.encode(self)
        
    @staticmethod
    def from_json(json_str: str) -> 'Goal':
        """Deserialize a goal from a JSON string."""
        return jsonpickle.decode(json_str)
    
class NegativeGoal(eqx.Module):
    """
    Converts a positive goal into a negative one. 
    """
    goals: Goal

    def __call__(self, model: Model, frequency: Frequency) -> jnp.ndarray:
        return -self.goal(model, frequency)
    
def make_negative_goals(cost_fn: Callable | Sequence) -> Callable | Sequence:
    """Converts Goals and sequences of Goals into log-likelihoods."""
    if isinstance(cost_fn, Sequence):
        return [NegativeGoal(c) if isinstance(c, Goal) else c for c in cost_fn]
    elif isinstance(cost_fn, Goal):
        return NegativeGoal(cost_fn)
    return cost_fn