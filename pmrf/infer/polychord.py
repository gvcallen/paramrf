from typing import Any, Callable, Dict
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jaxtyping import PyTree, Scalar, Array

MPI_AVAILABLE = False
try:
    from mpi4py import MPI
    import pypolychord
    from anesthetic import NestedSamples
    MPI_AVAILABLE = True
except ImportError:
    pass

from pmrf.infer.base import SamplingResult, AbstractCallableSampler

class PolyChord(AbstractCallableSampler):
    """
    A JAX-wrapped Nested Sampler using PolyChord.

    Acts as an adapter layer between JAX PyTrees and PolyChord's required flat 1D NumPy arrays.
    It automatically handles flattening and unflattening of complex parameter structures, 
    JIT-compiles the likelihood and prior transforms for performance, and bridges JAX operations 
    with PolyChord's host-based MPI sampling routines.

    Attributes
    ----------
    num_repeats : int | None
        The length of the slice sampling chain to generate a new live point. 
        If `None`, dynamically defaults to `5 * ndims`.
    nprior : int
        The number of prior samples to draw before clustering begins (default: -1).
    nfail : int
        The number of failed slice sampling steps before giving up (default: -1).
    do_clustering : bool
        Whether to use k-means clustering to handle multimodal posteriors (default: True).
    feedback : int
        The level of output written to stdout. 0=none, 1=standard, 2=detailed (default: 1).
    precision_criterion : float
        The stopping criterion based on the estimated evidence precision (default: 1e-3).
    logzero : float
        The numerical value used to represent log(0) (default: -1e30).
    boost_posterior : float
        Boost the number of live points near the peak to improve posterior samples (default: 0.0).
    posteriors : bool
        Whether to produce standard posterior output files (default: True).
    equals : bool
        Whether to output equally weighted posterior samples (default: True).
    cluster_posteriors : bool
        Whether to produce posterior output files for individual clusters (default: True).
    write_resume : bool
        Whether to continuously write resume files during the run (default: True).
    write_paramnames : bool
        Whether to generate a `.paramnames` file for post-processing tools (default: False).
    read_resume : bool
        Whether to attempt resuming from a previous partially completed run (default: True).
    write_stats : bool
        Whether to write run statistics to a `.stats` file (default: True).
    write_live : bool
        Whether to dump the current live points to disk (default: True).
    write_dead : bool
        Whether to record the dead points (the core nested sampling output) to disk (default: True).
    write_prior : bool
        Whether to write prior samples to disk (default: True).
    maximise : bool
        Whether to perform a maximization phase to find the exact MAP estimate (default: False).
    compression_factor : float
        The compression factor used for slice sampling (default: np.exp(-1.0)).
    synchronous : bool
        Whether to run MPI operations synchronously (default: True).
    base_dir : str
        The base directory path where all output files will be saved (default: "chains").
    file_root : str
        The root naming convention for all generated output files (default: "test").
    cluster_dir : str
        The directory name for cluster-specific outputs (default: "clusters").
    seed : int
        Random seed for the sampler. Uses time if set to -1 (default: -1).
    nlives : dict
        A dictionary mapping log-likelihood contours to the number of live points.
    paramnames : list of tuple
        A list of parameter names and LaTeX formatted names, e.g., `[("p1", r"\theta_1")]`.
        Must match the flatten dimension of `y0`.
    """
    nlive: int = -1
    num_repeats: int | None = None
    nprior: int = -1
    nfail: int = -1
    do_clustering: bool = True
    feedback: int = 1
    precision_criterion: float = 1e-3
    logzero: float = -1e30
    boost_posterior: float = 0.0
    posteriors: bool = True
    equals: bool = True
    cluster_posteriors: bool = True
    write_resume: bool = True
    write_paramnames: bool = False
    read_resume: bool = True
    write_stats: bool = True
    write_live: bool = True
    write_dead: bool = True
    write_prior: bool = True
    maximise: bool = False
    compression_factor: float = np.exp(-1.0)
    synchronous: bool = True
    base_dir: str = "chains"
    file_root: str = "test"
    cluster_dir: str = "clusters"
    seed: int = -1
    nlives: Dict[float, int] = eqx.field(static=True, default_factory=dict)
    paramnames: list[tuple[str, str]] | None = None

    @property
    def requires_hypercube(self):
        return True 
    
    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        prior_fn: Callable[[PyTree, Any], PyTree],
        y0: PyTree,
        init_samples: PyTree | None,
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
        max_steps: int | None,
    ) -> SamplingResult:
        if not MPI_AVAILABLE:
            raise ImportError("pypolychord, anesthetic and mpi4py must be installed to use the PolyChord sampler.")

        # 1. DERIVE GEOMETRY FROM y0
        flat_y0, reconstruct_fn = jax.flatten_util.ravel_pytree(y0)
        ndims = flat_y0.size

        max_ndead = max_steps if max_steps is not None else -1
        
        # Combine options
        options = options or {}
        base_kwargs = {
            'nlive': self.nlive,
            'num_repeats': self.num_repeats if self.num_repeats is not None else ndims*5,
            'nprior': self.nprior,
            'nfail': self.nfail,
            'do_clustering': self.do_clustering,
            'feedback': self.feedback,
            'precision_criterion': self.precision_criterion,
            'logzero': self.logzero,
            'max_ndead': max_ndead,
            'boost_posterior': self.boost_posterior,
            'posteriors': self.posteriors,
            'equals': self.equals,
            'cluster_posteriors': self.cluster_posteriors,
            'write_resume': self.write_resume,
            'write_paramnames': self.write_paramnames,
            'read_resume': self.read_resume,
            'write_stats': self.write_stats,
            'write_live': self.write_live,
            'write_dead': self.write_dead,
            'write_prior': self.write_prior,
            'maximise': self.maximise,
            'compression_factor': self.compression_factor,
            'synchronous': self.synchronous,
            'base_dir': self.base_dir,
            'file_root': self.file_root,
            'cluster_dir': self.cluster_dir,
            'seed': self.seed,
            'nlives': self.nlives,
            'paramnames': self.paramnames,
        }
        
        unknown_options = options.keys() - base_kwargs.keys() - {'nlive'}
        if unknown_options:
            raise ValueError(f"PolyChord sample got unknown options: {unknown_options}")
        
        if init_samples is not None:
            flatten_single = lambda x: jax.flatten_util.ravel_pytree(x)[0]
            cube_samples = jax.vmap(flatten_single)(init_samples)
            base_kwargs['cube_samples'] = np.array(cube_samples)
        
        base_kwargs.update(options)

        # 2. JIT-COMPILED BRIDGES
        @jax.jit
        def jitted_likelihood(flat_theta_jax):
            struct_theta = reconstruct_fn(flat_theta_jax)
            return loglikelihood_fn(struct_theta, args)

        @jax.jit
        def jitted_prior(flat_u_jax):
            struct_u = reconstruct_fn(flat_u_jax)
            struct_theta = prior_fn(struct_u, args)
            flat_theta, _ = jax.flatten_util.ravel_pytree(struct_theta)
            return flat_theta

        def polychord_likelihood(theta_np):
            logL = jitted_likelihood(jnp.asarray(theta_np))
            return float(logL), []

        def polychord_prior(u_np):
            return np.array(jitted_prior(jnp.asarray(u_np)))
        
        # Warmup / Test evaluation
        _dummy_prior = polychord_prior(0.5*np.ones(ndims))
        _dummy_logL = polychord_likelihood(_dummy_prior)

        # 3. EXECUTE POLYCHORD
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
        
        # Attempt to extract log evidence safely
        logevidence = None
        if hasattr(nested_samples, 'logZ'):
            try:
                logevidence = float(nested_samples.logZ())
            except Exception as e:
                logging.debug(f"Could not extract logZ from anesthetic NestedSamples: {e}")
        
        # 4. RESTRUCTURE RESULTS
        structured_samples = jax.vmap(reconstruct_fn)(jnp.array(samples))

        return SamplingResult(
            samples=structured_samples,
            loglikelihoods=loglikes,
            weights=weights,
            logevidence=logevidence,
            stats={'nested_samples': nested_samples, 'max_steps': max_steps}            
        )