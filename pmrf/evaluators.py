"""
Callables that evaluate a model over frequency.
"""
from __future__ import annotations
import re
from typing import Sequence, Literal, Any, Callable, TypeAlias
from abc import abstractmethod

import numpy as np
import equinox as eqx
import parax as prx
import jax
import jax.numpy as jnp
import distreqx.distributions as dist
import distreqx.bijectors as bij
from eqxpress import AbstractExpression, Stack, Method, Sum, Diagonal, Map, Index

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.losses import HingeLoss, RMSELoss
from pmrf.utils import field, unwrap, unwrap_self

class AbstractEvaluator(eqx.Module):
    """
    Abstract base class for callables that evaluate a model over frequency.

    Note that an evaluator should only depend on the model's parameter values,
    any not any additional metadata (bounds, distributions etc.).
    """
    @abstractmethod
    def __call__(self, model: Model, freq: Frequency, **kwargs) -> jnp.ndarray:
        """
        Evaluate the model response over the specified frequency range.

        Parameters
        ----------
        model : Model
            The model instance to evaluate.
        freq : Frequency
            The frequency object defining the evaluation points.
        **kwargs : dict
            Additional keyword arguments for the evaluation process.

        Returns
        -------
        jnp.ndarray
            The evaluated model response.
        """
        raise NotImplementedError
    
#: A type alias for the function signature of an evaluator.
EvaluatorFn: TypeAlias = Callable[[Model, Frequency], jnp.ndarray]

#: A type alias for "evaluator-like" objects, used as inputs to functions.
EvaluatorLike: TypeAlias = str | list[str] | EvaluatorFn | list[EvaluatorFn]

class Feature(AbstractEvaluator):
    """
    Extracts an RF feature using a string-based alias.
    
    Parses regex patterns to automatically route strings like 's11_db' or 
    'amplifier.y21_deg' to the appropriate Method and Indexing chain.
    """
    #: The underlying expression created that does the final feature extraction.
    expression: AbstractExpression

    def __init__(self, alias: str | Sequence[str] | list[AbstractEvaluator]):
        """Initialize the feature evalutor.

        Parameters
        ----------
        alias : str | Sequence[str] | list[Evaluator]
            A string alias, list of string aliases, or list of other evaluators for the feature.
        """
        super().__init__()

        # TODO should we integrate this class with the newer optics approach?
        
        # Lists of expressions
        if isinstance(alias, list) and all(isinstance(a, AbstractEvaluator) for a in alias):
            self.expression = Sum(alias)
            return

        # Sequences
        if not isinstance(alias, str) and isinstance(alias, Sequence):
            evaluators = tuple(Feature(a) for a in alias)
            self.expression = Stack(evaluators, axis=-1)
            return

        # Parse paths
        fields = alias.split('.')
        subattrs = ".".join(fields[:-1]) if len(fields) > 1 else ""
        local_alias = fields[-1]

        # Gamma/Tau special case (gamma = offdiagonal, tau=diagonal)
        if local_alias.startswith(('s_gamma', 's_tau')):
            is_gamma = 'gamma' in local_alias
            base_prop = local_alias.replace('s_gamma', 's', 1) if is_gamma else local_alias.replace('s_tau', 's', 1)
            path = f"{subattrs}.{base_prop}" if subattrs else base_prop
            
            base_evaluator = Method(path=path)
            
            if is_gamma:
                self.expression = Diagonal(base_evaluator)
            else:
                # We can't use OffDiagonal from eqxpress because we dont know the number of ports
                # (though we should really refactor eqxpress to support the buttom dynamically)
                self.expression = Map(
                    lambda mat: jax.vmap(lambda m: m[~jnp.eye(m.shape[-1], dtype=bool)])(mat),
                    base_evaluator, 
                )
            return

        # Parses port indices e.g. s11_db, y21
        # Matches a string starting with letters, exactly 2 digits, and an optional suffix
        rf_match = re.match(r'^([a-zA-Z]+)(\d)(\d)(_[a-zA-Z0-9_]+)?$', local_alias)
        if rf_match:
            prop_prefix, p1, p2, prop_suffix = rf_match.groups()
            prop_suffix = prop_suffix or ""
            
            path = f"{subattrs}.{prop_prefix}{prop_suffix}" if subattrs else f"{prop_prefix}{prop_suffix}"
            expression = Method(path=path)
            
            # Port indixing (keep leading dim i.e. frequency)
            indices = (slice(None), int(p1) - 1, int(p2) - 1)
            expression = Index(expression, indices)
            
        else:
            # Standard python attributes
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', local_alias):
                raise ValueError(f"Invalid feature alias format: '{alias}'")
                
            path = f"{subattrs}.{local_alias}" if subattrs else local_alias
            expression = Method(path=path)

        self.expression = expression

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        return self.expression(model, frequency, **kwargs)
     
    
class TargetLoss(AbstractEvaluator):
    """
    Computes a loss between a model prediction and some target.

    Parameters
    ----------
    predictor
        The predictor (e.g. another Evaluator) that extracts model features.
        Can be a function or a PyTree with optional parameters.
    target
        The fixed or 'true' target that the loss function should compare the prediction to.
    loss
        The loss function that takes (y_true, y_pred) and returns a loss metric.
        Can be a function or a PyTree with optional parameters.
        See :mod:`pmrf.losses` for common losses.
    """
    #: The active predictor instance.
    predictor: Callable[[Model, Frequency], jnp.ndarray]

    #: The fixed target data.
    target: np.ndarray = field(converter=np.asarray)
    
    #: The active loss function.
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        y_pred = self.predictor(model, frequency, **kwargs)
        return self.loss(self.target, y_pred)
    
    
class MarginalLogLikelihood(AbstractEvaluator):
    """
    Computes the log of the probability of observed data
    by conditioning a likelihood function on a model prediction
    while marginalizing out a potential discrepancy model.
    
    Includes a mapping from model "observation space" to "event space".
    By default, this defines the frequency axis as the probabilistic event,
    by moving it to the last axis before passing it to the likelihood/discrepancy.
    However, an bijective transform can be applied to model probability
    in an arbitrary latent space.

    Parameters
    ----------
    predictor
        The predictor (e.g. another Evaluator) that extracts model features.
        Can be a function or a PyTree with optional parameters.
    observed
        The observed data that the log probability will be computed of.
        Must have a shape that matches the shape of the predictor output.
    likelihood
        The likelihood function that takes the model prediction and returns the probability of observing some data.
        Can be a function or a PyTree with optional parameters.
        See :mod:`pmrf.likelihoods` for common likelihoods.
    discrepancy
        An optional discrepancy model to cater for model misspecification.
        Can be a function or a PyTree with optional parameters.
        See :class:`pmrf.discrepancy_models` for common discrepancy models.
    use_orthogonal_discrepancy
        Whether or not the discrepancy callable accepts a key-word argument "orthogonal_projection"
        which defines the model's orthogonal sub-space. Used for gaussian processes.
    event_transform
        A bijective transform that maps from "observation space" (predicted features) to "event space" (probability).
        Can be a bijector or None to use the default mapping (frequency as the event axis and independant real/imag).
    event_ndims
        The number of trailing event dimensions in event space to use as the event shape. Defaults to 1.
    """
    #: The active predictor instance.
    predictor: Callable[[Model, Frequency], jnp.ndarray]
    
    #: The observed data.
    observed: np.ndarray = field(converter=np.asarray)
    
    #: The active likelihood function.
    likelihood: Callable[[jnp.ndarray | dist.AbstractDistribution], dist.AbstractDistribution]

    #: The optional discrepancy model.
    discrepancy: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None
    
    #: Flag for orthogonal discrepancy.
    use_orthogonal_discrepancy: bool = field(default=False, static=True)

    #: The bijective event transform.
    event_transform: bij.AbstractBijector = field(default=None, static=True)

    #: The number of trailing event dimensions.
    event_ndims: int = field(default=1, static=True)
    
    def __post_init__(self):
        # Default mapping takes y_pred of shape e.g. (nfreq, nports, nports)
        # and maps to (nports, nports, nfreq) or (nports, nports, 2, nfreq) for complex.
        if self.event_ndims != 1:
            raise Exception("MarginalLogLikelihood currently only supports a single dependent event dimension")
        
        observed = unwrap(self.observed)
        if self.event_transform is None:
            ndims = observed.ndim
            if jnp.iscomplexobj(observed):
                perm = tuple(range(1, ndims + 1)) + (0,)
                self.event_transform = bij.Chain([bij.Transpose(perm), bij.Inverse(bij.R2ToComplex())])
            else:
                perm = tuple(range(1, ndims)) + (0,)
                self.event_transform = bij.Chain([bij.Transpose(perm)])
        
    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        observed = self.observed
        # Get the distribution over obs_event and the actual observed event
        obs_dist = self.predictive_distribution(model, frequency, **kwargs)
        obs_event = self.event_transform.forward(observed)
        batch_ndims = obs_event.ndim - self.event_ndims
        
        # We evaluate the log prob `batch_ndims` many times
        def eval_log_prob(d, x):
            return d.log_prob(x)
        mapped_log_prob = eval_log_prob
        for _ in range(batch_ndims):
            mapped_log_prob = eqx.filter_vmap(mapped_log_prob)
        
        log_probs = mapped_log_prob(obs_dist, obs_event)
        return jnp.sum(log_probs)
    
    def predictive_distribution(self, model: Model, frequency: Frequency, **kwargs) -> dist.AbstractDistribution:
        """
        Returns the full predictive distribution of an observed event for a given model.
        
        The returned distribution is in event space. To draw a sample from this distribution in
        observation space, see :meth:`MarginalLogLikelihood.sample_observation`.
        """
        self = unwrap(self)
        
        def event_fn(m, f):
            y_pred = self.predictor(m, f, **kwargs)
            return self.event_transform.forward(y_pred)
        
        discrepancy_kwargs = {}
        if self.use_orthogonal_discrepancy:
            jitter = 1e-12
            jac_dict = model.func_jacobian(event_fn, frequency)
            
            # J_b : shape (..., N, P)  # e.g., (nports, nports, nfreq, P)
            J_b = jnp.stack(tuple(jac_dict.values()), axis=-1)
            
            N, P = J_b.shape[-2], J_b.shape[-1]
            I = jnp.eye(N)
            
            J_b_T = jnp.swapaxes(J_b, -1, -2)
            JT_J = J_b_T @ J_b
            JT_J_stable = JT_J + jitter * jnp.eye(P)
            
            # JAX automatically vectorizes linalg.inv over leading dimensions
            JT_J_inv = jnp.linalg.inv(JT_J_stable) # Shape: (..., P, P)
            discrepancy_kwargs['orthogonal_projection'] = I - (J_b @ JT_J_inv @ J_b_T)
        
        pred_event = event_fn(model, frequency)
        if self.discrepancy is not None:
            pred_event = self.discrepancy(pred_event, frequency.f_scaled, **discrepancy_kwargs)
            
        # 4. Apply measurement noise likelihood (adds noise to the GP covariance)
        return self.likelihood(pred_event)
        
    @unwrap_self
    def sample_observation(self, key: jax.Array, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        """
        Returns a sample from the predictive distribution in observation space.
        """
        obs_dist = self.predictive_distribution(model, frequency, **kwargs)
        obs_event = self.event_transform.forward(self.observed)        
        
        # Get the actual shape of the batch dimensions
        batch_shape = obs_event.shape[:-self.event_ndims]
        batch_ndims = len(batch_shape)
        
        num_elements = np.prod(batch_shape) if batch_shape else 1
        keys = jax.random.split(key, num_elements)
        keys = keys.reshape(*batch_shape, -1) 

        def sample_one(d, k):
            return d.sample(jnp.squeeze(k))
        mapped_sample_fn = sample_one
        for _ in range(batch_ndims):
            mapped_sample_fn = eqx.filter_vmap(mapped_sample_fn)
        
        event_sample = mapped_sample_fn(obs_dist, keys)
        data_sample = self.event_transform.inverse(event_sample)
        return data_sample
    
    
class GibbsMarginalLogLikelihood(AbstractEvaluator):
    """
    Computes a generalized log-posterior (the Gibbs measure) using a loss function 
    instead of a strict generative likelihood.
    
    This is used for Generalized Bayesian Inference (GBI). It supports conditioning 
    a physical model prediction while marginalizing out a potential discrepancy model 
    using an Expected Loss framework.

    Parameters
    ----------
    predictor
        The predictor (e.g. another Evaluator) that extracts model features.
        Can be a function or a PyTree with optional parameters.
    observed
        The observed data that the loss will be computed against.
        Must have a shape that matches the shape of the predictor output.
    loss
        The loss function that takes (y_true, y_pred) and returns a loss metric.
        Can be a function or a PyTree with optional parameters.
    temperature
        The inverse-weight (temperature) of the Gibbs measure. 
        Higher temperatures create wider, less confident posteriors.
    discrepancy
        (experimental) An optional discrepancy model to cater for model misspecification.
    use_orthogonal_discrepancy
        (experimental) Whether or not the discrepancy callable accepts a key-word argument "orthogonal_projection"
        which defines the model's orthogonal sub-space. Used for gaussian processes.
    event_transform
        A bijective transform that maps from "observation space" (predicted features) to "event space".
    event_ndims
        The number of trailing event dimensions in event space to use as the event shape. Defaults to 1.
    """
    #: The active predictor instance.
    predictor: Callable[[Model, Frequency], jnp.ndarray]
    
    #: The observed data.
    observed: np.ndarray = field(converter=np.asarray)
    
    #: The active loss function.
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    
    #: The Gibbs measure temperature.
    temperature: float = field(default=1.0, static=True)

    #: The active discrepancy model.
    discrepancy: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None
    
    #: Flag for orthogonal discrepancy.
    use_orthogonal_discrepancy: bool = field(default=False, static=True)

    #: The bijective event transform.
    event_transform: bij.AbstractBijector = field(default=None, static=True)

    #: The number of event dimensions.
    event_ndims: int = field(default=1, static=True)
    
    def __post_init__(self):
        # Default mapping logic (matches standard MarginalLogLikelihood)
        if self.event_ndims != 1:
            raise Exception("GibbsMarginalLogLikelihood currently only supports a single dependent event dimension")
        
        observed = unwrap(self.observed)
        if self.event_transform is None:
            ndims = observed.ndim
            if jnp.iscomplexobj(observed):
                perm = tuple(range(1, ndims + 1)) + (0,)
                self.event_transform = bij.Chain([bij.Transpose(perm), bij.Inverse(bij.R2ToComplex())])
            else:
                perm = tuple(range(1, ndims)) + (0,)
                self.event_transform = bij.Chain([bij.Transpose(perm)])
                
    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        def event_fn(m, f):
            y_pred = self.predictor(m, f, **kwargs)
            return self.event_transform.forward(y_pred)
            
        discrepancy_kwargs = {}
        if self.use_orthogonal_discrepancy and self.discrepancy is not None:
            jitter = 1e-12
            jac_dict = model.func_jacobian(event_fn, frequency)
            
            J_b = jnp.stack(tuple(jac_dict.values()), axis=-1)  # shape (..., N, P)
            N, P = J_b.shape[-2], J_b.shape[-1]
            I = jnp.eye(N)
            
            J_b_T = jnp.swapaxes(J_b, -1, -2)
            JT_J = J_b_T @ J_b
            JT_J_stable = JT_J + jitter * jnp.eye(P)
            
            JT_J_inv = jnp.linalg.inv(JT_J_stable)
            discrepancy_kwargs['orthogonal_projection'] = I - (J_b @ JT_J_inv @ J_b_T)
            
        pred_event = event_fn(model, frequency)
        variance_penalty = 0.0
        
        # Follows concept of "Expected Loss", where you penalize the loss
        # by the discrepancy variance. Not yet tested.
        if self.discrepancy is not None:
            pred_dist = self.discrepancy(pred_event, frequency.f_scaled, **discrepancy_kwargs)
            pred_event = pred_dist.mean()
            variance_penalty = jnp.sum(pred_dist.variance())
            
        obs_event = self.event_transform.forward(self.observed)
        base_loss = self.loss(obs_event, pred_event)
        expected_loss = base_loss + variance_penalty
        
        # Generalized log-posterior (Gibbs measure)
        return -(expected_loss / self.temperature)
        
class NegativeLogLikelihood(AbstractEvaluator):
    """
    Computes the negative of the log of the probability of observed data.
    
    Wrapper around :class:`pmrf.evaluators.MarginalLogLikelihood`
    that is useful for performing Maximum Likelihood Estimation.

    Parameters
    ----------
    mll
        The underlying marginal log likelihood.
    """
    #: The underlying MLL instance.
    mll: MarginalLogLikelihood | GibbsMarginalLogLikelihood

    def __call__(self, model: Model, frequency: Frequency, **kwargs) -> jnp.ndarray:
        return -self.mll(model, frequency, **kwargs)


class Goal(TargetLoss):
    """
    Computes a design goal using a hinge-based loss evaluator.
    """
    def __init__(
        self,
        feature: str | AbstractExpression,
        operator: Literal['<', '<=', '>', '>=', '==', '='] = '==',
        target: float | jnp.ndarray = 0.0,
        weight: float | jnp.ndarray = 1.0,
        mask: jnp.ndarray | None = None,
        loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = RMSELoss(),
        multioutput: str | Any = 'uniform_average'
    ):
        """
        Initializes the optimization goal.

        Parameters
        ----------
        feature : str or AbstractExpression
            The feature to be evaluated. If a string is provided, it is 
            automatically wrapped in a :class:`Feature` expression.
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
        loss : str or Any, optional
            The base loss function. Defaults to RMSE.
            See :mod:`pmrf.losses` for common losses.
        multioutput : str or Any, optional
            Defines how to aggregate losses across multiple outputs. 
            Default is 'uniform_average'.
        """        
        predictor = Feature(feature) if isinstance(feature, str) else feature
        target = np.asarray(target, dtype=float)
        loss = HingeLoss(
            operator=operator,
            weight=weight,
            mask=mask,
            base_loss=loss,
            multioutput=multioutput
        )
        
        super().__init__(predictor=predictor, target=target, loss=loss)
        
__all__ = [
    'AbstractEvaluator',
    'Feature',
    'TargetLoss',
    'MarginalLogLikelihood',
    'GibbsMarginalLogLikelihood',
    'NegativeLogLikelihood',
    'Goal',
    'EvaluatorFn',
    'EvaluatorLike',
]