from typing import Callable, Sequence, Any
import logging
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jax import flatten_util

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.optimize.goal import make_negative_goals
from pmrf.parameters import ParameterGroup

from jaxopt.base import StochasticSolver

def optimize_polychord(
    model: Model,
    cost: Callable[[Model, Frequency], jnp.ndarray] | Sequence,
    frequency: Frequency,
    *,
    logger: logging.Logger | None = None,
    **kwargs
) -> tuple[Model, Any]:
    """
    Uses PolyChord nested sampling as a global optimizer.
    
    Translates frequentist cost functions (e.g., Goals) into log-likelihoods,
    runs Bayesian inference, and extracts the maximum likelihood parameters.
    """
    from pmrf.infer import sample_polychord
    
    logger = logger or logging.getLogger(__name__)

    log_likelihood_fn = make_negative_goals(cost)

    posterior_dist, nested_samples = sample_polychord(
        model=model,
        log_likelihood=log_likelihood_fn,
        frequency=frequency,
        logger=logger,
        **kwargs
    )

    idx = np.argmax(nested_samples.logL.values)
    
    model_param_names = model.flat_param_names()
    x_map = jnp.array([nested_samples[name].values[idx] for name in model_param_names])

    # 4. Safely reconstruct the model at the peak likelihood
    params_tree, static_tree = model.partition()
    _, unravel_fn = flatten_util.ravel_pytree(params_tree)
    optimized_model: Model = eqx.combine(unravel_fn(x_map), static_tree)

    # 5. Create Parameter Groups and attach the posterior
    param_group = ParameterGroup(model_param_names, posterior_dist)
    optimized_model = optimized_model.with_param_groups(param_group)

    return optimized_model, nested_samples