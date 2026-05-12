import logging
from typing import Any, Callable, Optional

import jax
from jaxtyping import PyTree, Array, Scalar
import jax.numpy as jnp
import equinox as eqx
import blackjax

from pmrf.infer.base import AbstractJointSampler, AbstractSplitSampler, SampleResults

class NUTS(AbstractJointSampler):
    """
    No-U-Turn Sampler (NUTS) using the BlackJAX backend.
    
    Automatically handles Stan-style window adaptation for the diagonal 
    inverse mass matrix and step size.
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
    ) -> tuple[SampleResults, PyTree]:
        if max_steps is None:
            raise ValueError("BlackJAX requires a static `max_steps` integer for jax.lax.scan.")
        if init_samples is not None:
            raise ValueError("BlackJAX `NUTS` does not yet support initial samples.")

        # 1. Create a pure scalar logprob function for the MCMC integrator
        def logprob_fn(x):
            return logposterior_fn(x, args)

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
        
        fn_values = eval_vmap(trace_state.position)

        # 6. Construct the standard payload
        payload = SampleResults(
            samples=trace_state.position,
            fn_values=fn_values,
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

    def run(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = 1000,
        **kwargs,
    ) -> tuple[SampleResults, PyTree]:
        if max_steps is None:
            raise ValueError("BlackJAX requires a static `max_steps` integer for jax.lax.scan.")
        if init_samples is not None:
            raise ValueError("BlackJAX `NUTS` does not yet support initial samples.")        

        # 1. Create a pure scalar logprob function
        def logprob_fn(x):
            return logposterior_fn(x, args)

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
        
        fn_values = eval_vmap(trace_state.position)

        payload = SampleResults(
            samples=trace_state.position,
            fn_values=fn_values,
        )

        return payload, trace_info
    

class NSS(AbstractSplitSampler):
    """
    (experimental) Nested Slice Sampler (NSS) using the BlackJAX backend.
    """
    num_delete: int = eqx.field(static=True)
    num_inner_steps: int = eqx.field(static=True)
    logZ_convergence: float = eqx.field(static=True, default=1e-3)

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
    ) -> tuple[SampleResults, PyTree]:
        logging.warning("BlackJAX NSS posterior may be truncated")

        if init_samples is None:
            raise ValueError("NSS requires `init_samples` (a batch of particles) to initialize.")
        if not hasattr(blackjax, 'nss'):
            raise ImportError("`nss` not found in `blackjax`. Make sure the relevant handley-lab fork is installed via e.g. `pip install git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta`.")

        # 1. Standardize functions for BlackJAX
        def logprior(y):
            return logprior_fn(y)
            
        def loglikelihood(y):
            return loglikelihood_fn(y, args)

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

        # 4. Weight Calculation (Dead Points)
        num_live = jax.tree_util.tree_leaves(init_samples)[0].shape[0]
        iters = jnp.arange(actual_steps)
        
        # Shrinking prior volume for dead points
        log_X = - (iters * self.num_delete) / num_live
        log_dX = log_X + jnp.log1p(-jnp.exp(-self.num_delete / num_live))
        
        dead_ll = infos.loglikelihood
        dead_unnorm_log_weights = dead_ll + (log_dX - jnp.log(self.num_delete))[:, None]

        # 5. Weight Calculation (Live Points)
        # The remaining prior volume is distributed equally among the remaining live points
        log_X_final = - (actual_steps * self.num_delete) / num_live
        live_ll = final_state.loglikelihood
        live_unnorm_log_weights = live_ll + log_X_final - jnp.log(num_live)

        # 6. Flatten Arrays to 1D Streams
        def flatten_batch(x):
            return x.reshape(-1, *x.shape[2:])

        flat_dead_samples = jax.tree_util.tree_map(flatten_batch, infos.particles)
        flat_dead_ll = flatten_batch(dead_ll)
        flat_dead_unnorm_weights = flatten_batch(dead_unnorm_log_weights)

        # Extract live samples
        live_samples = final_state.particles

        # Concatenate dead and live into the full posterior history
        def concat_dead_live(dead, live):
            return jnp.concatenate([dead, live], axis=0)

        all_samples = jax.tree_util.tree_map(concat_dead_live, flat_dead_samples, live_samples)
        all_ll = concat_dead_live(flat_dead_ll, live_ll)
        all_unnorm_log_weights = concat_dead_live(flat_dead_unnorm_weights, live_unnorm_log_weights)

        # 7. Log-Evidence and Error Calculation
        # Use BlackJAX's internal integrator state for exact matching
        logevidence = jnp.logaddexp(final_state.logZ, final_state.logZ_live)
        
        # Normalize the combined weights against the total evidence
        weights = jnp.exp(all_unnorm_log_weights - logevidence)

        # Calculate Skilling's Information H ≈ \sum(W_i * log(L_i)) - logZ 
        H = jnp.sum(weights * all_ll) - logevidence
        logevidence_error = jnp.sqrt(jnp.maximum(H, 0.0) / num_live)

        # 8. Final Payload
        payload = SampleResults(
            samples=all_samples,
            fn_values=all_ll,
            weights=weights,
            logevidence=logevidence,
            logevidence_error=logevidence_error
        )

        return payload, infos