import logging
from typing import Any, Callable, Optional

import jax
from jaxtyping import PyTree, Array, Scalar
import jax.numpy as jnp
import equinox as eqx
import blackjax

from pmrf.infer.backend.base import AbstractJointSampler, AbstractSplitSampler, SamplerPayload

class NUTS(AbstractJointSampler):
    """
    No-U-Turn Sampler (NUTS) using the BlackJAX backend.
    
    Automatically handles Stan-style window adaptation for the diagonal 
    inverse mass matrix and step size.
    """
    num_warmup: int = eqx.field(static=True, default=1000)
    target_acceptance_rate: float = eqx.field(static=True, default=0.8)

    def sample(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = 1000,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        if max_steps is None:
            raise ValueError("BlackJAX requires a static `max_steps` integer for jax.lax.scan.")
        if init_samples is None:
            raise ValueError("BlackJAX `NUTS` does not yet support initial samples.")

        # 1. Create a pure scalar logprob function for the MCMC integrator
        def logprob_fn(x):
            res = logposterior_fn(x, args)
            return res[0] if has_aux else res

        warmup_key, sample_key = jax.random.split(key)

        # 2. Run Window Adaptation (Warmup)
        logging.info(f"Running BlackJAX NUTS warmup ({self.num_warmup} steps)...")
        adapt = blackjax.window_adaptation(
            blackjax.nuts, 
            logprob_fn, 
            target_acceptance_rate=self.target_acceptance_rate,
            **kwargs
        )
        (last_state, parameters), _ = adapt.run(warmup_key, y0, num_steps=self.num_warmup)

        # 3. Build the static Kernel
        kernel = blackjax.nuts(logprob_fn, **parameters).step

        # 4. Define and execute the sampling loop
        def inference_loop(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        logging.info(f"Running BlackJAX NUTS sampling ({max_steps} steps)...")
        keys = jax.random.split(sample_key, max_steps)
        _, (trace_state, trace_info) = jax.lax.scan(inference_loop, last_state, keys)

        # 5. Post-process to recover Exact Log-Probs and Aux Data
        # We vmap over the trajectory of positions to get the final payload
        def eval_fn(y):
            return logposterior_fn(y, args)
            
        eval_vmap = jax.vmap(eval_fn)
        
        if has_aux:
            fn_values, auxes = eval_vmap(trace_state.position)
        else:
            fn_values = eval_vmap(trace_state.position)
            auxes = None

        # 6. Construct the standard payload
        payload = SamplerPayload(
            samples=trace_state.position,
            fn_values=fn_values,
            weights=None,
            auxes=auxes
        )

        return payload, trace_info


class HMC(AbstractJointSampler):
    """
    Hamiltonian Monte Carlo (HMC) using the BlackJAX backend.
    
    Requires a static number of integration steps. Automatically adapts 
    the step size and mass matrix.
    """
    num_warmup: int = eqx.field(static=True, default=1000)
    target_acceptance_rate: float = eqx.field(static=True, default=0.8)
    num_integration_steps: int = eqx.field(static=True, default=30)

    def sample(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = 1000,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        if max_steps is None:
            raise ValueError("BlackJAX requires a static `max_steps` integer for jax.lax.scan.")
        if init_samples is None:
            raise ValueError("BlackJAX `NUTS` does not yet support initial samples.")        

        # 1. Create a pure scalar logprob function
        def logprob_fn(x):
            res = logposterior_fn(x, args)
            return res[0] if has_aux else res

        warmup_key, sample_key = jax.random.split(key)

        # 2. Run Window Adaptation (Warmup)
        logging.info(f"Running BlackJAX HMC warmup ({self.num_warmup} steps)...")
        adapt = blackjax.window_adaptation(
            blackjax.hmc, 
            logprob_fn, 
            target_acceptance_rate=self.target_acceptance_rate,
            num_integration_steps=self.num_integration_steps,
            **kwargs
        )
        (last_state, parameters), _ = adapt.run(warmup_key, y0, num_steps=self.num_warmup)

        # 3. Build the static Kernel
        kernel = blackjax.hmc(logprob_fn, **parameters).step

        # 4. Define and execute the sampling loop
        def inference_loop(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        logging.info(f"Running BlackJAX HMC sampling ({max_steps} steps)...")
        keys = jax.random.split(sample_key, max_steps)
        _, (trace_state, trace_info) = jax.lax.scan(inference_loop, last_state, keys)

        # 5. Post-process to recover Exact Log-Probs and Aux Data
        def eval_fn(y):
            return logposterior_fn(y, args)
            
        eval_vmap = jax.vmap(eval_fn)
        
        if has_aux:
            fn_values, auxes = eval_vmap(trace_state.position)
        else:
            fn_values = eval_vmap(trace_state.position)
            auxes = None

        payload = SamplerPayload(
            samples=trace_state.position,
            fn_values=fn_values,
            weights=None,
            auxes=auxes
        )

        return payload, trace_info
    

class NSS(AbstractSplitSampler):
    """
    (experimental) Nested Slice Sampler (NSS) using the BlackJAX backend.
    """
    num_delete: int = eqx.field(static=True)
    num_inner_steps: int = eqx.field(static=True)
    logZ_convergence: float = eqx.field(static=True, default=1e-3)

    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: PyTree = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        if init_samples is None:
            raise ValueError("NSS requires `init_samples` (a batch of particles) to initialize.")
        if not hasattr(blackjax, 'nss'):
            raise ImportError("`nss` not found in `blackjax`. Make sure the relevant contrib package is installed.")

        # 1. Standardize functions for BlackJAX
        # Blackjax NSS kernel expects logprior_fn(pos) and loglikelihood_fn(pos)
        def logprior(y):
            return logprior_fn(y)
            
        def loglikelihood(y):
            res = loglikelihood_fn(y, args)
            return res[0] if has_aux else res

        kernel = blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps,
            **kwargs,
        )

        # 2. Initialization
        state = jax.jit(kernel.init)(init_samples)

        # 3. Step execution
        # Nested sampling often runs until convergence (logZ_live - logZ < tol).
        # If max_steps is provided, we use lax.scan for performance. 
        # If None, we use a Python loop (on the host) to check convergence.
        
        @jax.jit
        def step_fn(current_state, rng_key):
            return kernel.step(rng_key, current_state)

        if max_steps is not None:
            logging.info(f"Running NSS for fixed {max_steps} steps...")
            keys = jax.random.split(key, max_steps)
            
            def scan_step(carry, k):
                s, i = step_fn(carry, k)
                return s, (s, i)
            
            final_state, (trajectory, infos) = jax.lax.scan(scan_step, state, keys)
            actual_steps = max_steps
        else:
            logging.info("Running NSS until logZ convergence...")
            trajectory_list, infos_list = [], []
            curr_state, curr_key = state, key
            
            while True:
                curr_key, subkey = jax.random.split(curr_key)
                curr_state, info = step_fn(curr_state, subkey)
                
                trajectory_list.append(curr_state)
                infos_list.append(info)
                
                # Dynamic convergence check (host-side)
                delta_logZ = curr_state.logZ_live - curr_state.logZ
                if delta_logZ < self.logZ_convergence:
                    break
            
            final_state = curr_state
            trajectory = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trajectory_list)
            infos = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *infos_list)
            actual_steps = len(trajectory_list)

        # 4. Weight Calculation
        # Nested sampling computes weights based on the 'dead' particles
        num_live = jax.tree_util.tree_leaves(init_samples)[0].shape[0]
        iters = jnp.arange(actual_steps)
        
        # Standard Skilling evidence accumulation math
        log_X = - (iters * self.num_delete) / num_live
        log_dX = log_X + jnp.log1p(-jnp.exp(-self.num_delete / num_live))
        
        # Handle shape alignment for batched likelihoods
        ll = infos.loglikelihood
        log_weights = ll + (log_dX - jnp.log(self.num_delete))[:, None]
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

        # 5. Recovery of Aux Data (Post-process via vmap)
        if has_aux:
            def eval_aux(y):
                _, aux = loglikelihood_fn(y, args)
                return aux
            auxes = jax.vmap(eval_aux)(infos.particles)
        else:
            auxes = None

        # 6. Final Payload
        payload = SamplerPayload(
            samples=infos.particles,
            fn_values=infos.loglikelihood,
            weights=weights,
            auxes=auxes
        )

        # Add NSS-specific evidence diagnostics to metrics
        metrics = {
            "logZ": final_state.logZ,
            "logZ_live": final_state.logZ_live,
            "actual_steps": actual_steps,
            "info": infos
        }

        return payload, metrics