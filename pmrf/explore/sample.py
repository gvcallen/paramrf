import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.core import Model, Frequency, Evaluator
from pmrf.constants import EvaluatorLike
from pmrf.evaluators import Alias
from pmrf.explore.samplers import AbstractSampler
from pmrf.explore.result import ExploreResult


def sample(
    model: Model,
    engine: AbstractSampler,
    N: int = 100,
    *,
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
    engine : AbstractSampler
        The sampling engine/algorithm to use.
    N : int, default=100
        The total number of samples to generate (budget).
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
        features = Alias('s')
    elif not isinstance(features, Evaluator): 
        features = Alias(features)

    d = model.num_flat_params
    
    def eval_fn(U_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Closure to map hypercube proposals to physical params and evaluate."""
        return _evaluate_batch(model, U_batch, frequency, features, **kwargs)

    # 1. Initialize the solver state
    options = {"target_N": N}
    key, init_key = jr.split(key)
    state = engine.init(eval_fn, d, init_key, options)
    
    # 2. Standardized Execution Loop
    while not engine.terminate(state, N):
        key, step_key = jr.split(key)
        state = engine.step(eval_fn, d, state, step_key, options)

    # 3. Truncate if batching pushed us slightly over the budget
    thetas = state.sampled_params
    extracted_features = state.sampled_features
    if len(thetas) > N:
        thetas = thetas[:N]
        extracted_features = extracted_features[:N]

    # 4. Package the array of parameters back into a cleanly batched JAX PyTree
    batched_models = jax.vmap(model.with_params)(thetas)
    
    return ExploreResult(
        model=model, # Leave the original continuous model untouched
        frequency=frequency,
        sampled_models=batched_models, 
        sampled_features=extracted_features,
        history=state.backend_state
    )


def _evaluate_batch(model: Model, U: jnp.ndarray, frequency: Frequency | None, features: Evaluator, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper to map hypercube proposals to physical params and evaluate features."""
    def eval_single(u):
        flat_params = model.flat_params()
        theta = jnp.array([p.distribution.icdf(u_i) for p, u_i in zip(flat_params, u)])
        
        m_sampled = model.with_params(theta)
        feat_val = features(m_sampled, frequency, **kwargs) if frequency else features(m_sampled, **kwargs)
        return theta, feat_val

    return jax.vmap(eval_single)(U)