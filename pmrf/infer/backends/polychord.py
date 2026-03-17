import logging
from typing import Callable, Sequence, Any
import numpy as np
import jax.numpy as jnp

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.utils import time_string
from pmrf.infer.problem import InferenceProblem
from pmrf.distributions import Distribution

def sample_polychord(
    model: Model,
    log_likelihood: Callable[[Model, Frequency], jnp.ndarray] | Sequence,
    frequency: Frequency,
    *,
    logger: logging.Logger | None = None,
    nlive_factor: int = 25,
    **kwargs
) -> tuple[Distribution, Any]:
    """
    Executes PolyChord nested sampling and returns the posterior distribution.
    """
    logger = logger or logging.getLogger(__name__)
    problem = InferenceProblem(model, log_likelihood, frequency)

    import pypolychord
    from pmrf.distributions import AnestheticDistribution

    num_params = len(problem.param_names)
    kwargs.setdefault('nlive', nlive_factor * num_params)
    
    dot_param_names = [name.replace('_', '.') for name in problem.param_names]
    labeled_param_names = np.array([[n, dn] for n, dn in zip(problem.param_names, dot_param_names)])
    
    def log_likelihood_np(theta_np: np.ndarray):
        return float(problem.log_likelihood_fn(jnp.array(theta_np))), []

    def prior_np(u_np: np.ndarray):
        return np.array(problem.icdf_fn(jnp.array(u_np)), dtype=np.float64)
    
    # Compilation dry-run
    _ = log_likelihood_np(prior_np(0.5 * np.ones(num_params)))

    def dumper(_live, _dead, _logweights, logZ, _logZerr):
        logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')

    logger.info(f'PolyChord sampling started at {time_string()}...')
    nested_samples = pypolychord.run(
        log_likelihood_np, num_params, dumper=dumper, prior=prior_np, 
        paramnames=labeled_param_names, **kwargs
    )
    logger.info(f'PolyChord finished at {time_string()}')
    
    return AnestheticDistribution(nested_samples, problem.param_names), nested_samples