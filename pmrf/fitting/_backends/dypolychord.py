import jax.numpy as jnp
import numpy as np

from pmrf.fitting.bayesian import BayesianFitter, BayesianContext
from pmrf.fitting._backends.anesthetic import AnestheticResults
from pmrf.util import time_string

class dyPolyChordFitter(BayesianFitter):
    """
    dyPolyChord: Dynamic nested sampling using ``dyPolyChord``.
    """
    def _run_algorithm(self, ctx: BayesianContext, *, best_param_method='maximum-likelihood', nlive_init_factor=None, nlive_factor=None, dynamic_goal=1.0, **kwargs) -> AnestheticResults:
        """
        Executes the dynamic nested sampling run.

        Parameters
        ----------
        ctx : BayesianContext
            The Bayesian fitting context.
        best_param_method : str, optional, default='maximum-likelihood'
            Method to extract point estimates ('maximum-likelihood' or 'mean').
        nlive_init_factor : int, optional
            Factor for the initial number of live points (`ninit = factor * n_params`).
            Defaults to 5.
        nlive_factor : int, optional
            Factor for the target number of live points (`nlive_const = factor * n_params`).
            Defaults to 25.
        dynamic_goal : float, optional, default=1.0
            Dynamic goal (0 for evidence, 1 for posterior, or in between).
        **kwargs
            Additional arguments passed to ``dyPolyChord.run_dypolychord``.

        Returns
        -------
        AnestheticResults
            The fit results containing the nested samples.
        """
        # Dynamic imports to avoid hard dependencies
        import dyPolyChord
        import dyPolyChord.pypolychord_utils
        from anesthetic import read_chains
        
        # 1. Setup Configuration
        nlive_init_factor = nlive_init_factor if nlive_init_factor is not None else 5
        nlive_factor = nlive_factor if nlive_factor is not None else 25
        
        num_params = ctx.num_params
        ninit = nlive_init_factor * num_params
        nlive_const = nlive_factor * num_params
        
        if nlive_const <= ninit:
            self.logger.warning("nlive_const should typically be > ninit for dynamic runs. Adjusting ninit to be smaller.")
            ninit = max(nlive_const // 2, 2)
        
        # 2. Path and Settings Management
        # dyPolyChord expects a dictionary for the underlying PolyChord settings
        settings_dict = kwargs.pop('settings_dict', {}).copy()
        
        base_dir = f'{ctx.output_path}/chains' if ctx.output_path else 'chains'
        file_root = ctx.output_root or 'dypolychord_run'
        
        settings_dict['base_dir'] = base_dir
        settings_dict['file_root'] = file_root
        settings_dict['do_clustering'] = True
        settings_dict['read_resume'] = False

        # 3. Generate Functions
        x0 = np.array(ctx.model.flat_param_values())
        
        # Convert JAX functions to NumPy for PolyChord
        loglikelihood_fn = ctx.make_log_likelihood_fn(as_numpy=True)
        prior_fn = ctx.make_prior_transform_fn(as_numpy=True)
        
        # 4. Execute dyPolyChord
        # We use the utility wrapper provided by dyPolyChord to package the likelihood/prior
        run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
            loglikelihood_fn, prior_fn, num_params
        )

        self.logger.info(f'Fitting {num_params} parameters with dyPolyChord.')
        self.logger.info(f'Dynamic Goal: {dynamic_goal} | ninit: {ninit} | nlive_const: {nlive_const}')
        self.logger.info(f'Output: {base_dir}/{file_root}')
        self.logger.info(f'Started at {time_string()}')

        dyPolyChord.run_dypolychord(
            run_func,
            dynamic_goal,
            settings_dict_in=settings_dict,
            ninit=ninit,
            nlive_const=nlive_const,
            **kwargs
        )

        self.logger.info(f'Finished at {time_string()}')

        # 5. Process Results
        # Read chains using anesthetic
        root_path = f"{base_dir}/{file_root}"
        nested_samples = read_chains(root_path)

        # Apply parameter names manually as they aren't always preserved in the .txt output
        # unless written explicitly by the wrapper (which RunPyPolyChord does not do by default)
        param_names = ctx.combined_param_names()
        
        # Ensure we don't overwrite internal anesthetic columns
        limit = min(len(param_names), len(nested_samples.columns))
        mapper = {old: new for old, new in zip(nested_samples.columns[:limit], param_names)}
        nested_samples.rename(columns=mapper, inplace=True)
        
        # Set LaTeX labels for plotting
        dot_param_names = [name.replace('_', '.') for name in param_names]
        for name, dot_name in zip(param_names, dot_param_names):
            nested_samples.set_label(name, f'\\theta_{{{dot_name}}}')

        # 6. Extract Best Fit
        for i, param_name in enumerate(param_names[0:-ctx.num_likelihood_params]):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                # Use logL column to find the best sample
                idx = nested_samples.logL.idxmax()
                x0[i] = nested_samples.loc[idx, param_name]
            else:
                self.logger.warning("Unknown best parameter method. Skipping update.")
        
        fitted_model = ctx.model.with_params(x0)
        
        return AnestheticResults(fitted_model=fitted_model, solver_results=nested_samples)