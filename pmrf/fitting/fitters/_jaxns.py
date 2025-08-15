import jax
import time
import jax.numpy as jnp
import numpyro.distributions as dist

from pmrf.fitting._bayesian import BayesianFitter
from pmrf.fitting.results import AnestheticResults

class JAXNSFitter(BayesianFitter):
    """
    A fitter that uses the jaxns nested sampler.
    """
    def run(self, best_param_method: str = 'maximum-likelihood', num_live_points: int = None, termination_frac: float = 0.01, max_samples: float = 1e6, seed: int = 0) -> AnestheticResults:
        """
        Runs the jaxns nested sampler.

        Args:
            best_param_method (str): Method to determine the best-fit parameters from the posterior. 
                                     Options are 'maximum-likelihood' or 'mean'. Defaults to 'maximum-likelihood'.
            num_live_points (int): The number of live points to use for the nested sampling run. 
                                   If None, defaults to 50 * number of parameters.
            termination_frac (float): The termination condition for the sampler, defined as the fraction 
                                      of the remaining live evidence. Defaults to 0.01.
            max_samples (float): The maximum number of samples to generate before stopping. Defaults to 1e6.
            seed (int): The random seed for JAX's pseudo-random number generator. Defaults to 0.

        Returns:
            AnestheticResults: An object containing the results of the fitting process, including the nested samples.
        """
        import jaxns
        from anesthetic import NestedSamples
        
        start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)

        # --- 1. Parameter and Function Setup ---
        param_names = self._flat_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = {name: f'\\theta_{{{name_replaced}}}' for name, name_replaced in zip(param_names, dot_param_names)}
        priors = [param.prior for param in params]
        
        x0 = self.model.flat_params()
        loglikelihood_fn = self._make_log_likelihood_fn()
        prior_fn = self._make_prior_transform_fn()

        if num_live_points is None:
            num_live_points = 25 * len(param_names)

        self.logger.info(f'Fitting for {len(param_names)} parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        self.logger.info(f"Starting nested sampling with {num_live_points} live points...")

        model = jaxns.Model(prior_model=prior_model, log_likelihood=log_likelihood)
        
        ns_sampler = jaxns.NestedSampler(
            log_likelihood=loglikelihood_fn,
            prior_transform=prior_fn,
            num_live_points=num_live_points,
        )

        # run_numerical_integration handles the sampling loop and termination condition
        results, state = ns_sampler.run_numerical_integration(
            rng_key,
            term_cond=jaxns.TerminationCondition(live_log_evidence_frac=termination_frac),
            max_samples=int(max_samples),
            collect_samples=True,
            show_progress_bar=True
        )

        # --- 4. Finalize and Package Results ---
        self.logger.info("Finalizing results...")
        nested_samples = NestedSamples(
            data=results['samples'],
            columns=param_names,
            logL=results['log_L'],
            logL_birth=results['log_L_birth'],
            labels=labeled_param_names,
        )

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Sampling finished in {total_time:.2f} seconds.")
        
        # jaxns directly provides the log evidence (logZ) and its error
        logZ = results['log_Z']
        logZ_err = results['log_Z_err']
        self.logger.info(f"Final logZ = {logZ:.2f} +/- {logZ_err:.2f}")
        
        # --- 5. Update Model with Best-Fit Parameters ---
        model_param_names = list(self.model.flat_param_names())
        for i, param_name in enumerate(model_param_names):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping.")
                
        return AnestheticResults(
            fitted_model=None,
            initial_model=self.model,
            frequency=self.frequency,
            measured=self.measured,
            features=self.feature_list,
            logger=self.logger,
            solver_results=nested_samples,
            solver_args=(),
            fitter_kwargs={'best_param_method': best_param_method, 'termination_frac': termination_frac}
        )