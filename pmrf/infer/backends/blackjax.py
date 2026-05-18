"""
BlackJAX inference wrappers.
"""

import logging
import tqdm
from typing import Any, Callable, Optional

import jax
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree, Array, Scalar
import jax.numpy as jnp
import equinox as eqx

try:
    import blackjax
except ImportError:
    pass

from pmrf.infer.base import AbstractJointSampler, AbstractSplitSampler, SampleResult

# TODO we could maybe generalize the "basic" samplers like NUTS and HMC
# into a single MCMC wrapper similar to optimistix?

class NUTS(AbstractJointSampler):
    """
    No-U-Turn Sampler (NUTS) in JAX.

    Wrapper around :class:`blackjax.nuts`.
    
    Automatically handles Stan-style window adaptation for the diagonal 
    inverse mass matrix and step size.

    Parameters
    ----------
    num_warmup : int, default=1000
        Number of warmup steps for window adaptation.
    target_acceptance_rate : float, default=0.8
        Target acceptance rate for step size adaptation.
    """
    num_warmup: int = eqx.field(static=True, default=1000)
    target_acceptance_rate: float = eqx.field(static=True, default=0.8)

    def run(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = 1000,
        **kwargs,
    ) -> tuple[SampleResult, PyTree]:
        if max_steps is None:
            raise ValueError("BlackJAX requires a static `max_steps` integer for jax.lax.scan.")
        if init_samples is not None:
            raise ValueError("BlackJAX `NUTS` does not yet support initial samples.")

        def logprob_fn(x):
            return logposterior_fn(x, args)

        # MCMC warmup ("window adaptation")
        warmup_key, sample_key = jax.random.split(key)
        logging.info(f"Running BlackJAX NUTS warmup ({self.num_warmup} steps)...")
        adapt = blackjax.window_adaptation(
            blackjax.nuts, 
            logprob_fn, 
            target_acceptance_rate=self.target_acceptance_rate,
            **kwargs
        )
        (last_state, parameters), _ = adapt.run(warmup_key, y0, num_steps=self.num_warmup)

        # Build the kernel and loop
        kernel = blackjax.nuts(logprob_fn, **parameters).step
        def step_fn(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        logging.info(f"Running BlackJAX NUTS sampling ({max_steps} steps)...")
        keys = jax.random.split(sample_key, max_steps)
        _, (trace_state, trace_info) = jax.lax.scan(step_fn, last_state, keys)

        fn_values = jax.vmap(logprob_fn)(trace_state.position)

        result = SampleResult(
            samples=trace_state.position,
            fn_values=fn_values,
        )
        return result, trace_info


class HMC(AbstractJointSampler):
    """
    Hamiltonian Monte Carlo (HMC) in JAX.

    Wrapper around :class:`blackjax.hmc`.
    
    Requires a static number of integration steps. Automatically adapts 
    the step size and mass matrix.

    Parameters
    ----------
    num_warmup : int, default=1000
        Number of warmup steps for window adaptation.
    target_acceptance_rate : float, default=0.8
        Target acceptance rate for step size adaptation.
    num_integration_steps : int, default=30
        Number of integration steps per transition.
    """
    num_warmup: int = eqx.field(static=True, default=1000)
    target_acceptance_rate: float = eqx.field(static=True, default=0.8)
    num_integration_steps: int = eqx.field(static=True, default=30)

    def run(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = 1000,
        **kwargs,
    ) -> tuple[SampleResult, PyTree]:
        if max_steps is None:
            raise ValueError("BlackJAX requires a static `max_steps` integer for jax.lax.scan.")
        if init_samples is not None:
            raise ValueError("BlackJAX `NUTS` does not yet support initial samples.")        

        def logprob_fn(x):
            return logposterior_fn(x, args)

        # Rest of code is similar to NUTS above
        warmup_key, sample_key = jax.random.split(key)
        logging.info(f"Running BlackJAX HMC warmup ({self.num_warmup} steps)...")
        adapt = blackjax.window_adaptation(
            blackjax.hmc, 
            logprob_fn, 
            target_acceptance_rate=self.target_acceptance_rate,
            num_integration_steps=self.num_integration_steps,
            **kwargs
        )
        (last_state, parameters), _ = adapt.run(warmup_key, y0, num_steps=self.num_warmup)

        kernel = blackjax.hmc(logprob_fn, **parameters).step
        def step_fn(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        logging.info(f"Running BlackJAX HMC sampling ({max_steps} steps)...")
        keys = jax.random.split(sample_key, max_steps)
        _, (trace_state, trace_info) = jax.lax.scan(step_fn, last_state, keys)

        fn_values = jax.vmap(logprob_fn)(trace_state.position)

        result = SampleResult(
            samples=trace_state.position,
            fn_values=fn_values,
        )
        return result, trace_info
    

class NSS(AbstractSplitSampler):
    """
    (experimental) A Nested Slice Sampler (NSS) in JAX.

    A wrapper around BlackJAX's experimental NSS sampler.
    This requires a custom fork of BlackJAX, available via
    `pip install git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta`.

    Parameters
    ----------
    num_delete : int, optional
        Number of live points to delete per step and therefore vectorize over.
        Defaults to 0.1 x num_live if not provided.
    num_inner_steps : int
        The length of the short Markov chains used to update the live points.
        Defaults to 3 x dim if not provided.
    evidence_convergence : float, default=1e-3
        Threshold for evidence convergence when `max_steps` is None.
    block_size : int, optional
        The number of steps to execute on-device per block before checking convergence.
        Defaults to 100.
    """
    num_delete: int | None = eqx.field(static=True, default=None)
    num_inner_steps: int | None = eqx.field(static=True, default=None)
    evidence_convergence: float = eqx.field(static=True, default=1e-3)
    block_size: int | None = eqx.field(static=True, default=100)

    def run(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree], Scalar],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: PyTree = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SampleResult, PyTree]:
        if init_samples is None:
            raise ValueError("NSS requires `init_samples` (a batch of particles) to initialize.")
        if not hasattr(blackjax, 'nss'):
            raise ImportError("`nss` not found in `blackjax`...")
        
        from blackjax.ns.utils import log_weights, finalise, sample

        # Initialize settings
        num_live = jax.tree.leaves(init_samples)[0].shape[0]
        dim = ravel_pytree(y0)[0].size
        num_delete = self.num_delete if self.num_delete is not None else int(0.1 * num_live)
        num_inner_steps = self.num_inner_steps if self.num_inner_steps is not None else int(3 * dim)
        block_size = self.block_size
        if block_size is None:
            block_size = max_steps if max_steps is not None else 1
            
        logZ_convergence = jnp.log10(self.evidence_convergence)

        def logprior(y):
            return logprior_fn(y, args)
        def loglikelihood(y):
            return loglikelihood_fn(y, args)

        kernel = blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
            **kwargs,
        )

        state = jax.jit(kernel.init)(init_samples)
        
        @jax.jit
        def step_fn(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = kernel.step(subk, state)
            return (state, k), dead_point

        steps_taken = 0
        dead_blocks = []
        rng_key = key
        
        logging.info(f"Running NSS (block_size={block_size}, max_steps={max_steps})...")
        with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
            while True:
                converged = (state.logZ_live - state.logZ) < logZ_convergence
                budget_reached = (max_steps is not None) and (steps_taken >= max_steps)

                if converged or budget_reached:
                    if converged:
                        logging.info("NSS converged via logZ threshold.")
                    if budget_reached:
                        logging.info(f"NSS reached max_steps ceiling ({max_steps}).")
                    break

                current_block_size = block_size
                if max_steps is not None:
                    current_block_size = min(block_size, max_steps - steps_taken)

                @jax.jit
                def block_step_fn(carry):
                    next_carry, block_dead = jax.lax.scan(step_fn, init=carry, xs=None, length=current_block_size)
                    return next_carry, block_dead

                (state, rng_key), b_dead = block_step_fn((state, rng_key))
                flat_b_dead = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), b_dead)
                dead_blocks.append(flat_b_dead)
                
                steps_taken += current_block_size
                pbar.update(current_block_size * num_delete)

        # Cater for immediate convergence
        if dead_blocks:
            dead = [jax.tree.map(lambda *args: jnp.concatenate(args, axis=0), *dead_blocks)]
        else:
            dead = []

        rng_key, weight_key = jax.random.split(rng_key, 3)
        final_state = finalise(state, dead)
        log_w = log_weights(weight_key, final_state, shape=100)
        # unweighted_samples = sample(sample_key, final_state, shape=num_live)  # if we ever need unweighted samples
        logzs = jax.scipy.special.logsumexp(log_w, axis=0)

        logevidence = logzs.mean()
        logevidence_error = logzs.std()
        weights = jnp.exp(log_w).mean(axis=-1)
        samples = final_state.particles
        fn_values = final_state.loglikelihood

        result = SampleResult(
            samples=samples,
            fn_values=fn_values,
            weights=weights,
            logevidence=logevidence,
            logevidence_error=logevidence_error
        )

        return result, dead