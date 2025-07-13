import numpyro.distributions as dist
import jax
import time
import jax.numpy as jnp
import jax
import jax.numpy as jnp

from pmrf.fitting._bayesian import BayesianFitter
from pmrf.fitting.results import AnestheticResults

class BlackjaxNSFitter(BayesianFitter):
    """
    A fitter that uses the blackjax nested slice sampler (`blackjax.nss`).
    """
    def run(self, best_param_method = 'maximum-likelihood', num_live_points = None, num_delete: int = 5, num_inner_steps: int = 20, logZ_convergence: float = 10.0, seed: int = 0) -> AnestheticResults:
        import blackjax
        from anesthetic import NestedSamples
        from tqdm import tqdm
        
        start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)

        params = self._flat_params()
        param_names = [param.name for param in params]
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = {name: f'\\theta_{{{name_replaced}}}' for name, name_replaced in zip(param_names, dot_param_names)}
        priors = [param.prior for param in params]
        
        recon_fn, x0 = self._make_reconstruct_function(flat=True, return_params=True)
        loglikelihood_fn = self._make_loglikelihood_function(flat=True)
        logprior_fn = self._make_logprior_function(flat=True)

        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )

        rng_key, prior_key = jax.random.split(rng_key)
        num_live_points = num_live_points if num_live_points is not None else 25 * len(param_names)
        
        keys = jax.random.split(rng_key, len(priors))
        samples_per_param = []
        for i, prior in enumerate(priors):
            sample = prior.sample(keys[i], sample_shape=(num_live_points,))
            samples_per_param.append(jnp.reshape(sample, (num_live_points, -1)))
        initial_particles = jnp.concatenate(samples_per_param, axis=1)        

        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)

        live_points = init_fn(initial_particles)
        dead_points_list = []

        self.logger.info(f'Fitting for {len(param_names)} parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        self.logger.info(f"Starting nested sampling with {num_live_points} live points...")
        with tqdm(desc="Sampling", unit=" dead points") as pbar:
            while not live_points.logZ_live - live_points.logZ < -logZ_convergence:
                rng_key, step_key = jax.random.split(rng_key)
                live_points, dead_info = step_fn(step_key, live_points)
                dead_points_list.append(dead_info)
                pbar.update(num_delete)
                pbar.set_postfix(logZ=f"{live_points.logZ:.2f}")

        # 6. Finalize the run and package the results
        self.logger.info("Finalizing results...")
        final_dead_points = blackjax.ns.utils.finalise(live_points, dead_points_list)

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
        
        model_param_names = [param.name for param in self.initial_model.flat_params()]
        for i, param_name in enumerate(model_param_names):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
                
        return AnestheticResults(
            model=None,
            initial_model=self.initial_model,
            frequency=self.model_frequency,
            measured=self.measured,
            features=self.feature_list,
            logger=self.logger,
            solver_results=nested_samples,
            solver_args=(),
            fit_kwargs={'best_param_method': best_param_method}
        )