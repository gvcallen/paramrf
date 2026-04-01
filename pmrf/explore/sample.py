import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.core import Model, Frequency, Operator
from pmrf.constants import EvaluatorLike
from pmrf.evaluators import Feature
from pmrf.explore.samplers import AbstractSampler
from pmrf.explore.result import ExploreResult


def sample(
    model: Model,
    sampler: AbstractSampler,
    *,
    max_samples: int | None = None,
    frequency: Frequency | None = None,
    features: EvaluatorLike | None = None,
    key: jax.Array | None = None,
    **kwargs
) -> ExploreResult:
    """
    Explore the parameter space of a model using a specified sampling engine.

    This unified router executes the sampling algorithm using a standardized 
    state-machine loop (init, step, terminate), supporting both one-shot and 
    adaptive active learning strategies frictionlessly.
    
    Parameters
    ----------
    model : Model
        The parametric model to sample.
    sampler : AbstractSampler
        The sampling algorithm to use.
    max_samples : int | None, default=None
        The maximum number of samples to generate. For one-shot samplers, this 
        is the exact number generated. For adaptive samplers, this acts as a 
        computational budget. If None, adaptive samplers run until convergence.
    frequency : Frequency | None, default=None
        The frequency sweep for feature evaluation.
    features : EvaluatorLike | None, default=None
        The specific circuit features to extract.
    key : jax.Array | None, default=None
        JAX PRNG key for stochastic samplers.
    **kwargs
        Additional arguments passed to the underlying evaluators.

    Returns
    -------
    ExploreResult
        The comprehensive result object containing the original continuous model 
        and batched execution states.
    """
    if key is None:
        key = jr.PRNGKey(0)
        
    if features is None:
        features = Feature('s')
    elif not isinstance(features, Operator): 
        features = Feature(features)

    d = model.num_flat_params
    
    def eval_fn(U_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Closure to map hypercube proposals to physical params and evaluate."""
        return _evaluate_batch(model, U_batch, frequency, features, **kwargs)

    # 1. Initialize the solver state
    options = {"max_samples": max_samples}
    key, init_key = jr.split(key)
    state = sampler.init(eval_fn, d, init_key, options)
    
    # 2. Standardized Execution Loop
    while not sampler.terminate(state, max_samples):
        key, step_key = jr.split(key)
        state = sampler.step(eval_fn, d, state, step_key, options)

    # 3. Truncate if batching pushed us slightly over the budget
    thetas = state.sampled_params
    extracted_features = state.sampled_features
    if max_samples is not None and len(thetas) > max_samples:
        thetas = thetas[:max_samples]
        extracted_features = extracted_features[:max_samples]

    # 4. Package the array of parameters back into a cleanly batched JAX PyTree
    batched_models = jax.vmap(model.with_params)(thetas)
    
    return ExploreResult(
        model=model, # Leave the original continuous model untouched
        frequency=frequency,
        sampled_models=batched_models, 
        sampled_features=extracted_features,
        history=state.backend_state
    )


def _evaluate_batch(model: Model, U: jnp.ndarray, frequency: Frequency | None, features: Operator, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper to map hypercube proposals to physical params and evaluate features."""
    def eval_single(u):
        flat_params = model.flat_params()
        theta = jnp.array([p.distribution.icdf(u_i) for p, u_i in zip(flat_params, u)])
        
        m_sampled = model.with_params(theta)
        feat_val = features(m_sampled, frequency, **kwargs) if frequency else features(m_sampled, **kwargs)
        return theta, feat_val

    return jax.vmap(eval_single)(U)