import time
import jax
import jax.numpy as jnp
from jax.extend.backend import get_backend
from tqdm import tqdm

from pmrf.fitting.bayesian import BayesianFitter, BayesianContext
from pmrf.fitting._backends.anesthetic import AnestheticResults

class BlackJAXNSFitter(BayesianFitter):
    """
    BlackJAX: Nested slice sampling using ``blackjax.nss``.
    """
    def _run_algorithm(self, ctx: BayesianContext, best_param_method = 'maximum-likelihood', nlive_factor = None, num_delete = None, num_inner_steps = None, logZ_convergence: float = -3, seed: int = 0) -> AnestheticResults:
        """
        Executes the nested sampling process using BlackJAX.

        Parameters
        ----------
        ctx : BayesianContext
            The Bayesian fitting context containing model, priors, and likelihoods.
        best_param_method : str, optional, default='maximum-likelihood'
            The method used to determine the "fitted" model parameters from the posterior.
            Options are 'maximum-likelihood' (takes the sample with highest logL)
            or 'mean' (takes the weighted mean of the posterior samples).
        nlive_factor : int or None, optional
            Multiplier to determine the number of live points (`n_live`).
            `n_live` is calculated as `nlive_factor * num_params`.
            If None, defaults to 25.
        num_delete : int or None, optional
            Number of live points to delete and replace in each iteration.
            If None, defaults to 10% of `n_live` on CPU, or 50% of `n_live` on GPU/TPU
            to leverage vectorization.
        num_inner_steps : int or None, optional
            The number of MCMC steps taken to generate a new live point.
            If None, defaults to `3 * num_params`.
        logZ_convergence : float, optional, default=-3
            The convergence threshold for the log-evidence (`logZ`).
            The loop terminates when `logZ_live - logZ < logZ_convergence`.
        seed : int, optional, default=0
            Random seed for JAX RNG.

        Returns
        -------
        AnestheticResults
            The results object containing the fitted model and the `anesthetic.NestedSamples` object.
        """
        import blackjax
        from anesthetic import NestedSamples
        
        start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)

        param_names = ctx.combined_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = {name: f'\\theta_{{{name_replaced}}}' for name, name_replaced in zip(param_names, dot_param_names)}
        
        x0 = ctx.model.flat_param_values()
        prior_fn = jax.jit(ctx.make_prior_transform_fn())
        logprior_fn = jax.jit(ctx.make_log_prior_fn())
        loglikelihood_fn = jax.jit(ctx.make_log_likelihood_fn())

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
        
        model_param_names = list(ctx.model.flat_param_names())
        for i, param_name in enumerate(model_param_names):
            if best_param_method == 'mean':
                val_new = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                val_new = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
            x0 = x0.at[i].set(val_new)
            
        fitted_model = ctx.model.with_params(x0)

        return AnestheticResults(fitted_model=fitted_model, solver_results=nested_samples)