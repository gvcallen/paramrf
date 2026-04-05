"""
Model evaluators.
"""
from __future__ import annotations
import re
from typing import Sequence, Literal, Any, Callable

import jax
import jax.numpy as jnp
import parax as prx
from parax.op import Map, Stack, Method, Sum, Diagonal, Index
import distreqx.distributions as dist

from pmrf.core import Model, Frequency, Evaluator
from pmrf.losses import HingeLoss
from pmrf.utils import unwrap_base_distribution

class FeatureAlias(Evaluator):
    """
    Extracts an RF feature using a string-based alias.
    
    Parses regex patterns to automatically route strings like 's11_db' or 
    'amplifier.y21_deg' to the appropriate Method and Indexing chain.
    """
    op: prx.Operator

    def __init__(self, alias: str | Sequence[str] | list[prx.Operator]):
        super().__init__()
        
        # 1. Handle pre-instantiated Operator lists (Summation)
        if isinstance(alias, list) and all(isinstance(a, prx.Operator) for a in alias):
            self.op = Sum(alias)
            return

        # 2. Handle Sequences (Recursive Stacking)
        if not isinstance(alias, str) and isinstance(alias, Sequence):
            evaluators = tuple(FeatureAlias(a) for a in alias)
            self.op = Stack(operators=evaluators, axis=-1)
            return

        # 3. Parse submodel paths (e.g., "submodel.s11_db")
        fields = alias.split('.')
        subattrs = ".".join(fields[:-1]) if len(fields) > 1 else ""
        local_alias = fields[-1]

        # 4. Handle Special RF Port Groups (Gamma/Tau)
        if local_alias.startswith(('s_gamma', 's_tau')):
            is_gamma = 'gamma' in local_alias
            base_prop = local_alias.replace('s_gamma', 's', 1) if is_gamma else local_alias.replace('s_tau', 's', 1)
            path = f"{subattrs}.{base_prop}" if subattrs else base_prop
            base_evaluator = Method(path=path)
            
            if is_gamma:
                self.op = Diagonal(base_evaluator)
            else:
                # We assume OffDiagonal is defined as in our previous discussion
                # If n_ports isn't known here, we use a dynamic vmapped approach
                self.op = Map(
                    operator=base_evaluator, 
                    fn=lambda mat: jax.vmap(lambda m: m[~jnp.eye(m.shape[-1], dtype=bool)])(mat)
                )
            return

        # 5. Standard Regex Parsing (e.g., s11_db)
        match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', local_alias)
        if not match:
            raise ValueError(f"Invalid feature alias format: '{alias}'")

        prop_prefix, p1, p2, prop_suffix = match.groups()
        path = f"{subattrs}.{prop_prefix}{prop_suffix}" if subattrs else f"{prop_prefix}{prop_suffix}"
        
        node = Method(path=path)

        # 6. Apply Port Indexing if specified
        if p1 is not None and p2 is not None:
            # Slices lead freq dim + 0-indexed ports
            indices = (slice(None), int(p1) - 1, int(p2) - 1)
            node = Index(operator=node, indices=indices)
        
        self.op = node

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        return self.op(model, frequency, **kwargs)
  
    
class TargetLoss(Evaluator):
    """
    Computes a loss between a model prediction and a given target.
    """
    predictor: Callable[[Model, Frequency], jnp.ndarray]
    target: jnp.ndarray
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = prx.field(transparent=True)

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        y_pred = self.predictor(model, frequency, **kwargs)
        return self.loss(self.target, y_pred)
    
    
class DataLikelihood(Evaluator):
    """
    Computes the log probability of observing given data
    given a likelihood function conditioned on a model prediction.
    
    Allows for a probabilistic discrepancy model (e.g. a gaussian process),
    which returns the distribution over a model's prediction given that prediction.
    The discrepancy model accepts the model prediction and scaled frequency vector,
    and returns a distribution over the model's prediction.
    """
    predictor: Callable[[Model, Frequency], jnp.ndarray]
    data: jnp.ndarray
    likelihood: Callable[[jnp.ndarray], dist.AbstractDistribution]
    discrepancy: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        y_pred = self.predictor(model, frequency, **kwargs)
        
        if self.discrepancy is not None:
            y_pred = self.discrepancy(y_pred, frequency.f_scaled)
        
        likelihood = self.likelihood(y_pred)
        return jnp.sum(likelihood.log_prob(self.data))
    

class Goal(TargetLoss):
    """
    Computes a design goal using a hinge-based loss evaluator.
    """
    def __init__(
        self,
        feature: str | prx.Operator,
        operator: Literal['<', '<=', '>', '>=', '==', '='] = '==',
        target: float | jnp.ndarray = 0.0,
        weight: float | jnp.ndarray = 1.0,
        mask: jnp.ndarray | None = None,
        loss_fn: str | Any = 'rmse',
        multioutput: str | Any = 'uniform_average'
    ):
        super().__init__()
        
        self.predictor = FeatureAlias(feature) if isinstance(feature, str) else feature
        self.target = jnp.asarray(target)
        
        # We store the metric logic. HingeLoss should be a PyTree or static.
        self.loss = HingeLoss(
            operator=operator,
            weight=weight,
            mask=mask,
            base_loss_fn=loss_fn,
            multioutput=multioutput
        )
        
__all__ = [
    'FeatureAlias',
    'TargetLoss',
    'DataLikelihood',
    'Goal',
]