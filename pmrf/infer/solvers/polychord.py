"""
PolyChord inference wrappers.
"""

from typing import Any, Callable, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jaxtyping import PyTree, Array, Scalar

MPI_AVAILABLE = False
try:
    from mpi4py import MPI
    import pypolychord
    from anesthetic import NestedSamples
    MPI_AVAILABLE = True
except ImportError:
    pass

from pmrf.infer.base import AbstractHypercubeSampler, SampleResult

class PolyChord(AbstractHypercubeSampler):
    """
    The PolyChord Nested Sampler wrapped in a JAX interface.

    Acts as an adapter layer between JAX PyTrees and PolyChord's required flat 1D NumPy arrays.
    It automatically handles flattening and unflattening of complex parameter structures, 
    JIT-compiles the likelihood and prior transforms for performance, and bridges JAX operations 
    with PolyChord's host-based MPI sampling routines.

    All parameters default to None, deferring to PolyChord's internal defaults.

    Parameters
    ----------
    nlive : int | None
        The number of live points (PolyChord default: 25 * ndims).
    num_repeats : int | None
        The length of the slice sampling chain (PolyChord default: 5 * ndims).
    nprior : int | None
        The number of prior samples to draw before clustering begins.
    nfail : int | None
        The number of failed slice sampling steps before giving up.
    do_clustering : bool | None
        Whether to use k-means clustering to handle multimodal posteriors.
    feedback : int | None
        The level of output written to stdout. 0=none, 1=standard, 2=detailed.
    precision_criterion : float | None
        The stopping criterion based on the estimated evidence precision.
    logzero : float | None
        The numerical value used to represent log(0).
    boost_posterior : float | None
        Boost the number of live points near the peak to improve posterior samples.
    posteriors : bool | None
        Whether to produce standard posterior output files.
    equals : bool | None
        Whether to output equally weighted posterior samples.
    cluster_posteriors : bool | None
        Whether to produce posterior output files for individual clusters.
    write_resume : bool | None
        Whether to continuously write resume files during the run.
    write_paramnames : bool | None
        Whether to generate a `.paramnames` file for post-processing tools.
    read_resume : bool | None
        Whether to attempt resuming from a previous partially completed run.
    write_stats : bool | None
        Whether to write run statistics to a `.stats` file.
    write_live : bool | None
        Whether to dump the current live points to disk.
    write_dead : bool | None
        Whether to record the dead points (the core nested sampling output) to disk.
    write_prior : bool | None
        Whether to write prior samples to disk.
    maximise : bool | None
        Whether to perform a maximization phase to find the exact MAP estimate.
    compression_factor : float | None
        The compression factor used for slice sampling.
    synchronous : bool | None
        Whether to run MPI operations synchronously.
    base_dir : str | None
        The base directory path where all output files will be saved.
    file_root : str | None
        The root naming convention for all generated output files.
    cluster_dir : str | None
        The directory name for cluster-specific outputs.
    seed : int | None
        Random seed for the sampler. Uses time if set to -1.
    nlives : dict | None
        A dictionary mapping log-likelihood contours to the number of live points.
    paramnames : list of tuple | None
        A list of parameter names and LaTeX formatted names, e.g., `[("p1", r"\theta_1")]`.
    """
    nlive: int | None = None
    num_repeats: int | None = None
    nprior: int | None = None
    nfail: int | None = None
    do_clustering: bool | None = None
    feedback: int | None = None
    precision_criterion: float | None = None
    logzero: float | None = None
    boost_posterior: float | None = None
    posteriors: bool | None = None
    equals: bool | None = None
    cluster_posteriors: bool | None = None
    write_resume: bool | None = None
    write_paramnames: bool | None = None
    read_resume: bool | None = None
    write_stats: bool | None = None
    write_live: bool | None = None
    write_dead: bool | None = None
    write_prior: bool | None = None
    maximise: bool | None = None
    compression_factor: float | None = None
    synchronous: bool | None = None
    base_dir: str | None = None
    file_root: str | None = None
    cluster_dir: str | None = None
    seed: int | None = None
    nlives: Dict[float, int] | None = eqx.field(static=True, default=None)
    paramnames: list[tuple[str, str]] | None = None

    @property
    def requires_hypercube(self):
        return True 
    
    def run(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        prior_transform_fn: Callable[[PyTree, Any], PyTree],
        u0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_cube_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> SampleResult:
        if not MPI_AVAILABLE:
            raise ImportError("pypolychord, anesthetic and mpi4py must be installed to use the PolyChord sampler.")

        # 1. DERIVE GEOMETRY FROM y0
        flat_y0_cube, cube_reconstruct_fn = jax.flatten_util.ravel_pytree(u0)
        ndims = flat_y0_cube.size
        
        # Combine options: dynamically build kwargs without None values
        pc_settings = [
            'nlive', 'num_repeats', 'nprior', 'nfail', 'do_clustering',
            'feedback', 'precision_criterion', 'logzero', 'boost_posterior',
            'posteriors', 'equals', 'cluster_posteriors', 'write_resume',
            'write_paramnames', 'read_resume', 'write_stats', 'write_live',
            'write_dead', 'write_prior', 'maximise', 'compression_factor',
            'synchronous', 'base_dir', 'file_root', 'cluster_dir', 'seed',
            'nlives', 'paramnames'
        ]
        
        base_kwargs = {
            k: getattr(self, k) for k in pc_settings if getattr(self, k) is not None
        }

        if max_steps is not None:
            base_kwargs['max_ndead'] = max_steps
            
        if init_cube_samples is not None:
            flatten_single = lambda x: jax.flatten_util.ravel_pytree(x)[0]
            cube_samples = jax.vmap(flatten_single)(init_cube_samples)
            base_kwargs['cube_samples'] = np.array(cube_samples)
        
        base_kwargs.update(kwargs)

        @jax.jit
        def jitted_likelihood(flat_theta_jax):
            struct_theta = cube_reconstruct_fn(flat_theta_jax)
            return loglikelihood_fn(struct_theta, args)

        @jax.jit
        def jitted_prior(flat_u_jax):
            struct_u = cube_reconstruct_fn(flat_u_jax)
            struct_theta = prior_transform_fn(struct_u, args)
            flat_theta, _ = jax.flatten_util.ravel_pytree(struct_theta)
            return flat_theta

        def polychord_likelihood(theta_np):
            logL = jitted_likelihood(jnp.asarray(theta_np))
            return float(logL), []

        def polychord_prior(u_np):
            return np.array(jitted_prior(jnp.asarray(u_np)))
        
        _dummy_prior = polychord_prior(0.5 * np.ones(ndims))
        _dummy_logL = polychord_likelihood(_dummy_prior)
        
        nested_samples = pypolychord.run(
            loglikelihood=polychord_likelihood,
            nDims=ndims,
            nDerived=0,
            prior=polychord_prior,
            **base_kwargs,
        )
        
        loglikes = jnp.array(np.array(nested_samples['logL']))
        samples = jnp.array(np.array(nested_samples.iloc[:, :ndims]))
        weights = jnp.array(nested_samples.get_weights())
        
        # Sometimes nested_samples is an MCMCSamples object (presumably for short runs),
        # which doesn't have a logZ method
        try:
            logevidence = jnp.array(nested_samples.logZ())
            norm_weights = weights / jnp.sum(weights)
        
            H = jnp.sum(norm_weights * loglikes) - logevidence
            actual_nlive = base_kwargs.get('nlive', ndims * 25)
            logevidence_error = jnp.array((np.sqrt(max(H, 0.0) / actual_nlive)))
        except:
            logevidence = None
            logevidence_error = None
        
        structured_samples = jax.vmap(cube_reconstruct_fn)(jnp.array(samples))
        return SampleResult(
            samples=structured_samples,
            fn_values=loglikes,
            weights=weights,
            logevidence=logevidence,
            logevidence_error=logevidence_error,
            metrics=nested_samples,
        )