import numpyro.distributions as dist
import jax
import time
import jax.numpy as jnp
import jax
import jax.numpy as jnp

from pmrf.fitting._bayesian import BayesianFitter
from pmrf.fitting.results import AnestheticResults

class BlackJAXNSFitter(BayesianFitter):
    """
    A fitter that uses the blackjax nested slice sampler in `blackjax.nss`.
    """
    def run(self, best_param_method = 'maximum-likelihood', nlive_factor = None, num_delete = None, num_inner_steps = None, logZ_convergence: float = -3, seed: int = 0) -> AnestheticResults:
        import blackjax
        from anesthetic import NestedSamples
        from tqdm import tqdm
        # from jax.lib import xla_bridge
        from jax.extend.backend import get_backend
        
        start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)

        param_names = self._flat_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = {name: f'\\theta_{{{name_replaced}}}' for name, name_replaced in zip(param_names, dot_param_names)}
        
        x0 = self.initial_model.flat_params()
        prior_fn = jax.jit(self._make_prior_transform_fn())
        logprior_fn = jax.jit(self._make_log_prior_fn())
        loglikelihood_fn = jax.jit(self._make_log_likelihood_fn())

        d = len(param_names)
        nlive_factor = nlive_factor if nlive_factor is not None else 25
        n_live = nlive_factor * d
        if num_delete is None:
            if get_backend().platform == 'cpu':
                self.logger.info('Running BlackJAX on the CPU')
                num_delete = int(0.1*n_live)
            else:
                self.logger.info('Running BlackJAX on a GPU/TPU')
                num_delete = int(0.5*n_live)
        if num_inner_steps is None:
            num_inner_steps = 3 * d

        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )

        rng_key, prior_key = jax.random.split(rng_key)
        u = jax.random.uniform(prior_key, shape=(n_live, d))
        initial_particles = jax.vmap(prior_fn)(u)

        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)

        state = init_fn(initial_particles)
        dead_points_list = []

        self.logger.info(f'Fitting for {len(param_names)} parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        self.logger.info(f"Starting nested sampling with {n_live} live points and {num_delete} delete points...")
        with tqdm(desc="Sampling", unit=" dead points") as pbar:
            while not state.logZ_live - state.logZ < logZ_convergence:
                rng_key, step_key = jax.random.split(rng_key)
                state, dead_info = step_fn(step_key, state)
                dead_points_list.append(dead_info)
                pbar.update(num_delete)
                pbar.set_postfix(logZ=f"{state.logZ:.2f}")

        # 6. Finalize the run and package the results
        self.logger.info("Finalizing results...")
        final_dead_points = blackjax.ns.utils.finalise(state, dead_points_list)

        # Use anesthetic to easily calculate logZ and its error
        nested_samples = NestedSamples(
            data=final_dead_points.particles,
            columns=param_names,
            logL=final_dead_points.loglikelihood,
            logL_birth=final_dead_points.loglikelihood_birth,
            labels=labeled_param_names,
        )

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Sampling finished in {total_time:.2f} seconds.")
        self.logger.info(f"Final logZ = {nested_samples.logZ()}")
        
        model_param_names = list(self.initial_model.flat_param_names())
        for i, param_name in enumerate(model_param_names):
            if best_param_method == 'mean':
                val_new = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                val_new = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
            x0 = x0.at[i].set(val_new)
            
        fitted_model = self.initial_model.with_flat_params(x0)
                
        return AnestheticResults(
            initial_model=self.initial_model,
            fitted_model=fitted_model,
            settings=self._settings,
            solver_results=nested_samples,
            solver_args=(),
            fitter_kwargs={'best_param_method': best_param_method}
        )