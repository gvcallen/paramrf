import io
import time
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.backend import get_backend
from tqdm.auto import tqdm

from pmrf.fitting.bayesian import BayesianFitter
from pmrf.models.model import Model
from pmrf.parameters import ParameterGroup

class BlackJAXNSFitter(BayesianFitter):
    """
    BlackJAX: Nested slice sampling using ``blackjax.nss``.
    """
    def optimize(
        self, 
        target_features: jnp.ndarray, 
        *,
        fitted_params='maximum-likelihood', 
        nlive_factor=None, 
        num_delete=None, 
        num_inner_steps=None, 
        logZ_convergence: float = -3.0, 
        seed: int = 0,
        **kwargs
    ) -> tuple[Model, Any]:
        """
        Executes the nested sampling process using BlackJAX.
        """
        import blackjax
        from anesthetic import NestedSamples
        from pmrf.distributions import AnestheticDistribution
        
        start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)

        # 1. Parameter Names Mapping
        param_names = self.model.flat_param_names() + list(self.likelihood_params.keys())
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = {
            name: f'\\theta_{{{name_replaced}}}' 
            for name, name_replaced in zip(param_names, dot_param_names)
        }
        
        d = len(param_names)
        x0 = np.array(self.model.flat_param_values())

        # 2. Trigger Lazy Compilation & Extract Pure JAX Functions
        # We pass dummy arrays to force BaseRunner to compile the JAX graphs
        theta0 = self.model.flat_param_values()
        _ = self.log_prior(theta0)
        _ = self.log_likelihood(theta0, target_features)
        _ = self.icdf(theta0)

        # Grab the internal compiled JAX closures
        logprior_fn = self._log_prior_fn
        prior_fn = self._icdf_fn

        # Wrap the likelihood so it only expects theta (closing over target_features)
        def loglikelihood_fn(theta):
            return self._log_likelihood_fn(theta, target_features)

        # 3. Setup BlackJAX Configuration
        nlive_factor = nlive_factor if nlive_factor is not None else 25
        n_live = nlive_factor * d
        
        if num_delete is None:
            if get_backend().platform == 'cpu':
                self.logger.info('Running BlackJAX on the CPU')
                num_delete = int(0.1 * n_live)
            else:
                self.logger.info('Running BlackJAX on a GPU/TPU')
                num_delete = int(0.5 * n_live)
                
        if num_inner_steps is None:
            num_inner_steps = 3 * d

        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )

        # 4. Initialize Particles
        rng_key, prior_key = jax.random.split(rng_key)
        u = jax.random.uniform(prior_key, shape=(n_live, d))
        initial_particles = jax.vmap(prior_fn)(u)

        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)

        state = init_fn(initial_particles)
        dead_points_list = []

        # 5. Execute Sampling Loop
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
        
        # 7. Determine Best Parameters
        num_model_params = self.model.num_flat_params
        for i, param_name in enumerate(param_names[0:num_model_params]):
            if fitted_params == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif fitted_params == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
            
        # 8. Update Model Priors with Posterior
        model_param_names = self.model.flat_param_names()
        param_group = ParameterGroup(
            model_param_names, 
            AnestheticDistribution(nested_samples, model_param_names)
        )
        fitted_model = self.model.with_params(x0).with_param_groups(param_group)

        return fitted_model, nested_samples

    # --------------------------------------------------------------------------
    # HDF5 Serialization Hooks (Identical to PolyChord since both return NestedSamples)
    # --------------------------------------------------------------------------
    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        """
        Encodes anesthetic NestedSamples into a CSV stream for HDF5.
        """
        csv_str: str = results.to_csv()
        stream.write(csv_str.encode('utf-8'))

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        """
        Reconstructs anesthetic NestedSamples from a CSV stream.
        """
        from anesthetic import NestedSamples, read_csv
        csv_str = stream.read().decode('utf-8')
        return NestedSamples(read_csv(io.StringIO(csv_str)))