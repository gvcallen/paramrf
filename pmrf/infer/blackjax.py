import jax
import jax.numpy as jnp
import equinox as eqx
import blackjax
from typing import Any, Callable, Optional
from jaxtyping import PyTree, Scalar, Array

from pmrf.infer.base import MCMCSamplingResult, AbstractMCMCSampler, NestedSamplingResult, AbstractNestedSampler


def _run_blackjax_sampler(
    algo_init: Callable,
    loglikelihood_fn: Callable[[PyTree, Any], Scalar],
    logprior_fn: Callable[[PyTree, Any], Scalar],
    y0: PyTree,
    key: Array,
    args: PyTree[Any],
    num_warmup: int,
    num_samples: int,
    target_acceptance: float,
    is_mass_matrix_diagonal: bool,
    algo_kwargs: dict[str, Any]
):
    """
    Shared execution loop for BlackJAX algorithms. Handles window adaptation,
    memory-efficient inline log-likelihood calculation, and tuple unpacking.
    """
    def logdensity_fn(params: PyTree) -> Scalar:
        return loglikelihood_fn(params, args) + logprior_fn(params, args)

    warmup_key, sample_key = jax.random.split(key)

    # 1. Warmup (Window Adaptation)
    adapt = blackjax.window_adaptation(
        algo_init, 
        logdensity_fn,
        target_acceptance_rate=target_acceptance,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        **algo_kwargs
    )
    (state, parameters), _ = adapt.run(warmup_key, y0, num_steps=num_warmup)

    # 2. Kernel Setup
    kernel = algo_init(logdensity_fn, **parameters, **algo_kwargs).step

    # 3. Memory-Efficient Sampling Loop
    # By computing the log-likelihood *inside* the loop, we avoid a massive vmap
    # over the entire trajectory history, completely bypassing the OOM risk.
    def one_step(state, key):
        state, info = kernel(key, state)
        ll = loglikelihood_fn(state.position, args)
        return state, (state, info, ll)
    
    keys = jax.random.split(sample_key, num_samples)
    final_state, (trajectory, infos, loglikes) = jax.lax.scan(one_step, state, keys)

    return final_state, trajectory, infos, loglikes, parameters


class NUTS(AbstractMCMCSampler):
    """
    (experimental) No-U-Turn Sampler (NUTS) using the BlackJAX backend.
    """
    num_warmup: int  #: Number of warmup steps for window adaptation
    num_samples: int  #: Number of samples to draw post-warmup
    target_acceptance: float = 0.8  #: Target acceptance rate (default 0.8, up to 0.95+ for complex geometries)
    max_num_doublings: int = 10  #: Maximum depth of the tree expansion (caps at 2^10 steps per iteration)
    is_mass_matrix_diagonal: bool = True  #: If True, adapts a diagonal mass matrix. If False, adapts a dense mass matrix.

    requires_hypercube: bool = False

    @eqx.filter_jit
    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        logprior_fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
    ) -> MCMCSamplingResult:
        
        algo_kwargs = {"max_num_doublings": self.max_num_doublings}

        final_state, trajectory, infos, loglikes, parameters = _run_blackjax_sampler(
            algo_init=blackjax.nuts,
            loglikelihood_fn=loglikelihood_fn,
            logprior_fn=logprior_fn,
            y0=y0,
            key=key,
            args=args,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            target_acceptance=self.target_acceptance,
            is_mass_matrix_diagonal=self.is_mass_matrix_diagonal,
            algo_kwargs=algo_kwargs
        )

        # 4. Pack Results
        # Note: Casting (e.g., int(), float()) is removed to prevent ConcretizationTypeError inside JIT.
        stats = {
            "algo": "NUTS",
            "acceptance_probability": infos.acceptance_probability,
            "is_divergent": infos.is_divergent,
            "energy": infos.energy,
            "num_trajectory_expansions": infos.num_trajectory_expansions,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "total_divergences": jnp.sum(infos.is_divergent),
            "mean_acceptance_rate": jnp.mean(infos.acceptance_probability),
            "tuned_step_size": parameters.step_size,
        }
        
        # Include runtime options in the output stats for transparency
        stats.update(options)

        return MCMCSamplingResult(
            samples=trajectory.position,
            loglikelihoods=loglikes,
            final_state=final_state,
            stats=stats,
        )


class HMC(AbstractMCMCSampler):
    """
    (experimental) Hamiltonian Monte Carlo (HMC) using the BlackJAX backend.
    """
    num_warmup: int  #: Number of warmup steps for window adaptation
    num_samples: int  #: Number of samples to draw post-warmup
    num_integration_steps: int  #: Number of leapfrog integration steps per HMC iteration
    target_acceptance: float = 0.8  #: Target acceptance rate for step size adaptation (default 0.8)
    is_mass_matrix_diagonal: bool = True  #: If True, adapts a diagonal mass matrix. If False, adapts a dense mass matrix.

    requires_hypercube: bool = False

    @eqx.filter_jit
    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        logprior_fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
    ) -> MCMCSamplingResult:
        
        algo_kwargs = {"num_integration_steps": self.num_integration_steps}

        final_state, trajectory, infos, loglikes, parameters = _run_blackjax_sampler(
            algo_init=blackjax.hmc,
            loglikelihood_fn=loglikelihood_fn,
            logprior_fn=logprior_fn,
            y0=y0,
            key=key,
            args=args,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            target_acceptance=self.target_acceptance,
            is_mass_matrix_diagonal=self.is_mass_matrix_diagonal,
            algo_kwargs=algo_kwargs
        )

        # 4. Pack Results
        # Note: Casting (e.g., int(), float()) is removed to prevent ConcretizationTypeError inside JIT.
        stats = {
            "algo": "HMC",
            "acceptance_probability": infos.acceptance_probability,
            "is_divergent": infos.is_divergent,
            "energy": infos.energy,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_integration_steps": self.num_integration_steps,
            "total_divergences": jnp.sum(infos.is_divergent),
            "mean_acceptance_rate": jnp.mean(infos.acceptance_probability),
            "tuned_step_size": parameters.step_size,
        }

        # Include runtime options in the output stats for transparency
        stats.update(options)

        return MCMCSamplingResult(
            samples=trajectory.position,
            loglikelihoods=loglikes,
            final_state=final_state,
            stats=stats,
        )
    
class NSS(AbstractNestedSampler):
    """
    (experimental) Nested Slice Sampler (NSS) using the BlackJAX backend, 
    wrapped in a single-call execution interface.
    """
    
    num_iterations: int  #: Total number of nested sampling iterations to run (replaces while-loop)
    num_delete: int  #: Number of live points to delete and replace per iteration
    num_inner_steps: int  #: Number of slice sampling steps to take for each new live point

    logZ_convergence: float = 1e-3  #: Convergence threshold for the evidence integral (ΔlogZ)

    requires_hypercube: bool = False

    @eqx.filter_jit
    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        prior_fn: Callable[[PyTree, Any], PyTree] | Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        init_samples: Optional[PyTree],
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
    ) -> NestedSamplingResult:
        
        # BlackJAX NSS takes isolated prior and likelihood functions.
        # Since requires_hypercube is False, prior_fn is guaranteed to return a Scalar log-prior.
        def logprior(y: PyTree) -> Scalar:
            return prior_fn(y, args) # type: ignore
            
        def loglikelihood(y: PyTree) -> Scalar:
            return loglikelihood_fn(y, args)

        # 1. Kernel Setup
        kernel = blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps
        )

        # 2. Initialization
        state = kernel.init(init_samples)

        # 3. Sampling Loop
        def one_step(state, key):
            state, info = kernel.step(key, state)
            return state, (state, info)
        
        keys = jax.random.split(key, self.num_iterations)
        final_state, (trajectory, infos) = jax.lax.scan(one_step, state, keys)

        # 4. Convergence Check
        is_converged = (final_state.logZ_live - final_state.logZ) < self.logZ_convergence

        # 5. Reconstruct Posterior Weights
        # Extract the number of live points from the leading dimension of init_samples
        num_live = jax.tree_util.tree_leaves(init_samples)[0].shape[0]
        
        # Calculate the log-prior volume (log_X) at each iteration step
        iters = jnp.arange(self.num_iterations)
        log_X = - (iters * self.num_delete) / num_live
        
        # Calculate the log-prior volume difference (shrinkage shell) for the current contour
        log_dX = log_X + jnp.log1p(-jnp.exp(-self.num_delete / num_live))
        
        # Compute log-weights for each dead point: log(W) = log(L) + log(dX)
        # We subtract jnp.log(num_delete) to distribute the volume evenly among all points deleted in this step.
        ll = jnp.atleast_2d(infos.loglikelihood)
        if ll.shape[0] != log_dX.shape[0]: 
            ll = ll.T # Catch edge cases depending on how BlackJAX stacks
        log_weights = ll + (log_dX - jnp.log(self.num_delete))[:, None]
        
        # Normalize the log-weights so that the final linear weights sum exactly to 1.0
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

        stats = {
            "algo": "NSS",
            "logZ_live_final": final_state.logZ_live,
            "is_converged": is_converged,
            "num_iterations": self.num_iterations,
            "num_delete": self.num_delete,
            "num_inner_steps": self.num_inner_steps,
        }
        
        stats.update(options)

        return NestedSamplingResult(
            samples=infos.particles,  
            loglikelihoods=infos.loglikelihood,
            weights=weights,  # Reconstructed, normalized, and perfectly mapped to shape!
            logevidence=final_state.logZ,
            logevidence_err=None,
            final_state=final_state,
            aux=infos,  
            stats=stats,
        )