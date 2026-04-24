import jax
import jax.numpy as jnp
import equinox as eqx
import blackjax
import logging
from typing import Any, Callable, Optional
from jaxtyping import PyTree, Scalar, Array

from pmrf.infer.base import SamplingResult, AbstractCallableSampler


def _run_blackjax_sampler(
    algo_init: Callable,
    loglikelihood_fn: Callable[[PyTree, Any], Scalar],
    logprior_fn: Callable[[PyTree, Any], Scalar],
    y0: PyTree,
    key: Array,
    args: PyTree[Any],
    num_warmup: int,
    max_steps: Optional[int],
    target_acceptance: float,
    is_mass_matrix_diagonal: bool,
    algo_kwargs: dict[str, Any]
):
    """
    Shared execution loop for BlackJAX algorithms. Handles window adaptation,
    memory-efficient inline log-likelihood calculation, and JIT/Python loop branching.
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
    
    # Bake num_warmup into the closure so it acts statically for JIT
    @jax.jit
    def run_warmup(k, y):
        return adapt.run(k, y, num_steps=num_warmup)
        
    (state, parameters), _ = run_warmup(warmup_key, y0)

    # 2. Kernel Setup
    kernel = algo_init(logdensity_fn, **parameters, **algo_kwargs).step

    # 3. Inner JIT step
    @jax.jit
    def one_step(state, key):
        state, info = kernel(key, state)
        ll = loglikelihood_fn(state.position, args)
        return state, info, ll

    # 4. Branch Execution Loop
    if max_steps is not None:
        @jax.jit
        def scan_loop(init_state, keys):
            def scan_step(state, key):
                state, info, ll = one_step(state, key)
                return state, (state, info, ll)
            return jax.lax.scan(scan_step, init_state, keys)
            
        keys = jax.random.split(sample_key, max_steps)
        final_state, (trajectory, infos, loglikes) = scan_loop(state, keys)
        actual_steps = max_steps
        
    else:
        logging.info("MCMC max_steps is None. Running in infinite Python loop. Press Ctrl+C to interrupt and yield partial chains.")
        trajectory_list = []
        infos_list = []
        loglikes_list = []
        
        current_key = sample_key
        try:
            while True:
                current_key, subkey = jax.random.split(current_key)
                state, info, ll = one_step(state, subkey)
                
                trajectory_list.append(state)
                infos_list.append(info)
                loglikes_list.append(ll)
                
        except KeyboardInterrupt:
            logging.info(f"Sampling interrupted. Collected {len(trajectory_list)} samples.")
            
        if not trajectory_list:
            raise RuntimeError("Sampling was interrupted before any samples were collected.")
            
        # Stack the collected PyTrees into arrays
        trajectory = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trajectory_list)
        infos = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *infos_list)
        loglikes = jnp.stack(loglikes_list)
        final_state = state
        actual_steps = len(trajectory_list)

    return final_state, trajectory, infos, loglikes, parameters, actual_steps


class NUTS(AbstractCallableSampler):
    """
    (experimental) No-U-Turn Sampler (NUTS) using the BlackJAX backend.
    """
    target_acceptance: float = 0.8  #: Target acceptance rate (default 0.8)
    max_num_doublings: int = 10  #: Maximum depth of the tree expansion
    is_mass_matrix_diagonal: bool = True  #: If True, adapts a diagonal mass matrix.

    requires_hypercube: bool = False

    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        logprior_fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
        num_warmup: int,
        max_steps: Optional[int],
    ) -> SamplingResult:
        
        algo_kwargs = {"max_num_doublings": self.max_num_doublings}

        final_state, trajectory, infos, loglikes, parameters, actual_steps = _run_blackjax_sampler(
            algo_init=blackjax.nuts,
            loglikelihood_fn=loglikelihood_fn,
            logprior_fn=logprior_fn,
            y0=y0,
            key=key,
            args=args,
            num_warmup=num_warmup,
            max_steps=max_steps,
            target_acceptance=self.target_acceptance,
            is_mass_matrix_diagonal=self.is_mass_matrix_diagonal,
            algo_kwargs=algo_kwargs
        )

        stats = {
            "algo": "NUTS",
            "acceptance_probability": infos.acceptance_probability,
            "is_divergent": infos.is_divergent,
            "energy": infos.energy,
            "num_trajectory_expansions": infos.num_trajectory_expansions,
            "num_warmup": num_warmup,
            "max_steps": actual_steps,
            "total_divergences": jnp.sum(infos.is_divergent),
            "mean_acceptance_rate": jnp.mean(infos.acceptance_probability),
            "tuned_step_size": parameters.step_size,
        }
        stats.update(options)

        return SamplingResult(
            samples=trajectory.position,
            loglikelihoods=loglikes,
            final_state=final_state,
            stats=stats,
        )


class HMC(AbstractCallableSampler):
    """
    (experimental) Hamiltonian Monte Carlo (HMC) using the BlackJAX backend.
    """
    num_integration_steps: int  #: Number of leapfrog integration steps per HMC iteration
    target_acceptance: float = 0.8  #: Target acceptance rate for step size adaptation
    is_mass_matrix_diagonal: bool = True  #: If True, adapts a diagonal mass matrix.

    requires_hypercube: bool = False

    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        logprior_fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
        num_warmup: int,
        max_steps: Optional[int],
    ) -> SamplingResult:
        
        algo_kwargs = {"num_integration_steps": self.num_integration_steps}

        final_state, trajectory, infos, loglikes, parameters, actual_steps = _run_blackjax_sampler(
            algo_init=blackjax.hmc,
            loglikelihood_fn=loglikelihood_fn,
            logprior_fn=logprior_fn,
            y0=y0,
            key=key,
            args=args,
            num_warmup=num_warmup,
            max_steps=max_steps,
            target_acceptance=self.target_acceptance,
            is_mass_matrix_diagonal=self.is_mass_matrix_diagonal,
            algo_kwargs=algo_kwargs
        )

        stats = {
            "algo": "HMC",
            "acceptance_probability": infos.acceptance_probability,
            "is_divergent": infos.is_divergent,
            "energy": infos.energy,
            "num_warmup": num_warmup,
            "max_steps": actual_steps,
            "num_integration_steps": self.num_integration_steps,
            "total_divergences": jnp.sum(infos.is_divergent),
            "mean_acceptance_rate": jnp.mean(infos.acceptance_probability),
            "tuned_step_size": parameters.step_size,
        }
        stats.update(options)

        return SamplingResult(
            samples=trajectory.position,
            loglikelihoods=loglikes,
            final_state=final_state,
            stats=stats,
        )
    

class NSS(AbstractCallableSampler):
    """
    (experimental) Nested Slice Sampler (NSS) using the BlackJAX backend.
    """
    
    num_delete: int  #: Number of live points to delete and replace per iteration
    num_inner_steps: int  #: Number of slice sampling steps to take for each new live point

    logZ_convergence: float = 1e-3  #: Convergence threshold for the evidence integral (ΔlogZ)

    requires_hypercube: bool = False

    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        prior_fn: Callable[[PyTree, Any], PyTree] | Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        init_samples: Optional[PyTree],
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
        max_steps: Optional[int],
    ) -> SamplingResult:
        
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
        state = jax.jit(kernel.init)(init_samples)

        # 3. Inner JIT step
        @jax.jit
        def one_step(state, key):
            return kernel.step(key, state)
        
        # 4. Branch Execution Loop
        if max_steps is not None:
            @jax.jit
            def scan_loop(init_state, keys):
                def scan_step(state, key):
                    state, info = one_step(state, key)
                    return state, (state, info)
                return jax.lax.scan(scan_step, init_state, keys)
                
            keys = jax.random.split(key, max_steps)
            final_state, (trajectory, infos) = scan_loop(state, keys)
            
            is_converged = bool((final_state.logZ_live - final_state.logZ) < self.logZ_convergence)
            actual_steps = max_steps
            
        else:
            logging.info("NSS max_steps is None. Running Python loop until logZ_convergence is met.")
            trajectory_list = []
            infos_list = []
            
            current_key = key
            while True:
                current_key, subkey = jax.random.split(current_key)
                state, info = one_step(state, subkey)
                
                trajectory_list.append(state)
                infos_list.append(info)
                
                # Check dynamic convergence
                # We cast to float to pull the scalar out of JAX context and evaluate truthiness 
                is_converged = bool((state.logZ_live - state.logZ) < self.logZ_convergence)
                if is_converged:
                    logging.info(f"NSS Converged after {len(trajectory_list)} steps.")
                    break
                    
            trajectory = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trajectory_list)
            infos = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *infos_list)
            final_state = state
            actual_steps = len(trajectory_list)

        # 5. Reconstruct Posterior Weights dynamically using `actual_steps`
        num_live = jax.tree_util.tree_leaves(init_samples)[0].shape[0]
        iters = jnp.arange(actual_steps)
        log_X = - (iters * self.num_delete) / num_live
        log_dX = log_X + jnp.log1p(-jnp.exp(-self.num_delete / num_live))
        
        ll = jnp.atleast_2d(infos.loglikelihood)
        if ll.shape[0] != log_dX.shape[0]: 
            ll = ll.T 
            
        log_weights = ll + (log_dX - jnp.log(self.num_delete))[:, None]
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

        stats = {
            "algo": "NSS",
            "logZ_live_final": final_state.logZ_live,
            "is_converged": is_converged,
            "num_delete": self.num_delete,
            "num_inner_steps": self.num_inner_steps,
            "max_steps": actual_steps,
        }
        stats.update(options)

        return SamplingResult(
            samples=infos.particles,  
            loglikelihoods=infos.loglikelihood,
            weights=weights,
            logevidence=final_state.logZ,
            logevidence_err=None,
            final_state=final_state,
            aux=infos,  
            stats=stats,
        )