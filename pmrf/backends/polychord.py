import io
from typing import Any
import numpy as np
import jax.numpy as jnp

from pmrf.models.model import Model
from pmrf.util import time_string

def run_polychord(
    model: Model,
    log_likelihood_fn: callable,
    icdf_fn: callable,
    param_names: list[str],
    logger,
    output_path=None,
    output_root='polychord',
    fitted_params='maximum-likelihood',
    nlive_factor=25,
    **kwargs
) -> tuple[Model, Any]:
    """
    Shared backend function for executing PolyChord nested sampling.
    """
    # Dynamic imports for heavy external dependencies
    import pypolychord
    from pmrf.parameters import ParameterGroup
    from pmrf.distributions import AnestheticDistribution

    num_model_params = model.num_flat_params
    
    # 1. Setup PolyChord Configuration
    kwargs.setdefault('nlive', nlive_factor * len(param_names))

    if output_path is not None:
        kwargs.setdefault('base_dir', f'{output_path}/chains')
        kwargs.setdefault('file_root', output_root)
    
    # 2. Map Parameter Names
    dot_param_names = [name.replace('_', '.') for name in param_names]
    labeled_param_names = np.array([[n, dn] for n, dn in zip(param_names, dot_param_names)])
    
    # 3. Define Wrappers for Lazily Compiled JAX Functions
    def log_likelihood_np(x):
        return float(log_likelihood_fn(x))

    def prior_np(u):
        return np.array(icdf_fn(u))
    
    # Compile / dry-run the JAX functions
    theta0 = 0.5 * jnp.ones(len(param_names))
    _ = log_likelihood_np(theta0)
    _ = prior_np(theta0)

    dumper = lambda _live, _dead, _logweights, logZ, _logZerr: \
        logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')

    # 4. Execute Nested Sampling 
    logger.info(f'PolyChord started at {time_string()}')
    nested_samples = pypolychord.run(
        log_likelihood_np,
        len(param_names),
        dumper=dumper,
        prior=prior_np,
        paramnames=labeled_param_names,
        **kwargs
    )
    logger.info(f'PolyChord finished at {time_string()}')
    
    # 5. Determine Best Parameters (Only updating actual model params)
    x0 = np.array(model.flat_param_values())
    model_param_names = param_names[:num_model_params]
    
    for i, param_name in enumerate(model_param_names):
        if fitted_params == 'mean':
            x0[i] = nested_samples[param_name].mean()
        elif fitted_params == 'maximum-likelihood':
            idx = np.argmax(nested_samples.logL.values)
            x0[i] = nested_samples[param_name].values[idx]
            
    # 6. Update Model Priors with the full Posterior Distribution
    param_group = ParameterGroup(
        model_param_names, 
        AnestheticDistribution(nested_samples, model_param_names)
    )
    fitted_model = model.with_params(x0).with_param_groups(param_group)
    return fitted_model, nested_samples


class PolyChordIOMixin:
    """Shared I/O methods for encoding/decoding anesthetic NestedSamples."""
    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        samples = results
        csv_str: str = samples.to_csv()
        stream.write(csv_str.encode('utf-8'))

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        from anesthetic import NestedSamples, read_csv
        csv_str = stream.read().decode('utf-8')
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        return samples