"""
Extractors that evaluate a model across frequency and output an array.
"""
from __future__ import annotations
import re
from typing import Sequence, Literal, Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import parax as prx
from parax.op import Map, Stack, Method, Sum, Diagonal, Index
import distreqx.distributions as dist

from pmrf.core import Model, Frequency, Evaluator
from pmrf.losses import HingeLoss, LogMSELoss

class Feature(Evaluator):
    """
    Extracts an RF feature using a string-based alias.
    
    Parses regex patterns to automatically route strings like 's11_db' or 
    'amplifier.y21_deg' to the appropriate Method and Indexing chain.
    """
    #: The underlying operator created that does the final feature extraction.
    op: prx.Operator

    def __init__(self, alias: str | Sequence[str] | list[prx.Operator]):
        """Initialize the feature evalutor.

        Parameters
        ----------
        alias : str | Sequence[str] | list[prx.Operator]
            A string alias, list of string aliases, or list of other evaluators for the feature.
        """
        super().__init__()
        
        # 1. Handle pre-instantiated Operator lists (Summation)
        if isinstance(alias, list) and all(isinstance(a, prx.Operator) for a in alias):
            self.op = Sum(alias)
            return

        # 2. Handle Sequences (Recursive Stacking)
        if not isinstance(alias, str) and isinstance(alias, Sequence):
            evaluators = tuple(Feature(a) for a in alias)
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
    Computes a loss between a model prediction and some target.
    """
    #: The predictor (e.g. another Evaluator) that extracts model features.
    #: Can be a function or a PyTree with optional parameters.
    predictor: Callable[[Model, Frequency], jnp.ndarray]

    #: The fixed or 'true' target that the loss function should compare the prediction to.
    target: jnp.ndarray
    
    #: The loss function that takes (y_true, y_pred) and returns a loss metric.
    #: Can be a function or a PyTree with optional parameters.
    #: See :mod:`pmrf.losses` for common losses.
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = prx.field(transparent=True)

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        y_pred = self.predictor(model, frequency, **kwargs)
        return self.loss(self.target, y_pred)
    
    
class MarginalLogLikelihood(Evaluator):
    """
    Computes the log of the probability of observing some data
    by conditioning a likelihood function on a model prediction
    while marginalizing out a potential discrepancy model.
    
    Performs the mapping from "data space" to "event space".
    By default, this defines the frequency axis as the probabilistic event,
    by moving it to the last axis before passing it to the likelihood/discrepancy.
    Real and imaginary parts are also stacked appropriately.
    """
    #: The predictor (e.g. another Evaluator) that extracts model features.
    #: Can be a function or a PyTree with optional parameters.
    predictor: Callable[[Model, Frequency], jnp.ndarray]
    
    #: The fixed 'observed' data that the log probability will be computed of.
    data: jnp.ndarray
    
    #: The likelihood function that takes the model prediction and returns the probability of observing some data.
    #: Can be a function or a PyTree with optional parameters.
    #: See :mod:`pmrf.likelihoods` for common likelihoods.
    likelihood: Callable[[jnp.ndarray], dist.AbstractDistribution]

    #: An optional discrepancy model to cater for model misspecification.
    #: Can be a function or a PyTree with optional parameters.
    #: See :class:`pmrf.discrepancy_models` for common discrepancy models.
    discrepancy: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None

    #: A mapper to map the "data space" (predicted features) to an "event space" (probability).
    event_mapper: Callable[[jnp.ndarray], jnp.ndarray] | None = None

    #: The number of trailing event dimensions returned by the event mapped. Defaults to 1.
    event_ndims: int = prx.field(default=1, static=True)

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        if self.event_ndims != 1:
            raise Exception("MarginalLogLikelihood currently only supports a single event dimension")
        
        # Default mapping takes y_pred of shape e.g. (nfreq, nports, nports)
        # and maps to (nports, nports, nfreq) or (nports, nports, 2, nfreq) for complex.
        def default_event_map(y_pred):
            y_event = y_pred
            if jnp.iscomplexobj(y_event):
                y_event = jnp.stack([jnp.real(y_event), jnp.imag(y_event)], axis=-1)
            y_event = jnp.moveaxis(y_event, 0, -1)
            return y_event
        
        # Get pred_event and obs_event
        y_pred = self.predictor(model, frequency, **kwargs)
        mapper = self.event_mapper if self.event_mapper is not None else default_event_map
        pred_event = mapper(y_pred)
        obs_event = mapper(self.data)
        if self.discrepancy is not None:
            pred_event = self.discrepancy(pred_event, frequency.f_scaled)
        
        # Get the distribution over obs_event
        obs_dist = self.likelihood(pred_event)
        
        # Sum the log probs over the batch dimension
        batch_ndims = obs_event.ndim - self.event_ndims
        def eval_log_prob(d, x):
            return d.log_prob(x)
        mapped_log_prob = eval_log_prob
        for _ in range(batch_ndims):
            mapped_log_prob = eqx.filter_vmap(mapped_log_prob)
        log_probs = mapped_log_prob(obs_dist, obs_event)
        return jnp.sum(log_probs)


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
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = LogMSELoss(),
        multioutput: str | Any = 'uniform_average'
    ):
        """
        Initializes the optimization goal.

        Parameters
        ----------
        feature : str or prx.Operator
            The feature to be evaluated. If a string is provided, it is 
            automatically wrapped in a :class:`Feature` operator.
        operator : {'<', '<=', '>', '>=', '==', '='}, optional
            The relational operator defining the goal condition. 
            '==' and '=' are treated as equivalent (equality). 
            Default is '=='.
        target : float or jnp.ndarray, optional
            The target value or array of values for the goal. 
            Default is 0.0.
        weight : float or jnp.ndarray, optional
            A scaling factor applied to the computed loss. Can be a 
            scalar or an array for element-wise weighting. 
            Default is 1.0.
        mask : jnp.ndarray, optional
            A boolean or numerical mask used to include or exclude specific 
            data points (e.g., specific frequencies) from the loss calculation. 
            Default is None.
        loss_fn : str or Any, optional
            The base loss function. Defaults to LogMSE.
            See :mod:`pmrf.losses` for common losses.
        multioutput : str or Any, optional
            Defines how to aggregate losses across multiple outputs. 
            Default is 'uniform_average'.

        Attributes
        ----------
        predictor : prx.Operator
            The operator used to extract the feature from the model response.
        target : jnp.ndarray
            The processed target value(s) stored as a JAX array.
        loss : HingeLoss
            The internal evaluator that implements the hinge logic and 
            metric calculation.
        """        
        predictor = Feature(feature) if isinstance(feature, str) else feature
        target = jnp.asarray(target)
        loss = HingeLoss(
            operator=operator,
            weight=weight,
            mask=mask,
            base_loss_fn=loss_fn,
            multioutput=multioutput
        )
        
        super().__init__(predictor=predictor, target=target, loss=loss)
        
__all__ = [
    'Feature',
    'TargetLoss',
    'MarginalLogLikelihood',
    'Goal',
]