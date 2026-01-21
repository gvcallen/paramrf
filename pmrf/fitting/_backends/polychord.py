import jax.numpy as jnp
import numpy as np

from pmrf.fitting.bayesian import BayesianFitter, BayesianContext
from pmrf.fitting._backends.anesthetic import AnestheticResults
from pmrf._util import time_string
   
class PolyChordFitter(BayesianFitter):
    """
    Polychord fitter using ``pypolychord.run``.
    
    PolyChord has its own license available at https://github.com/PolyChord/PolyChordLite.
    """
    def _run(self, ctx: BayesianContext, *, best_param_method='maximum-likelihood', nlive_factor=25, **kwargs) -> AnestheticResults:
        """
        Executes the PolyChord nested sampling run.

        Parameters
        ----------
        ctx : BayesianContext
            The Bayesian fitting context containing model, priors, and likelihoods.
        best_param_method : str, optional, default='maximum-likelihood'
            The method used to determine the "fitted" model parameters from the posterior.
            Options are 'maximum-likelihood' (takes the sample with highest logL)
            or 'mean' (takes the weighted mean of the posterior samples).
        nlive_factor : int, optional, default=25
            Factor to multiply by the number of parameters to determine `nlive` (number of live points).
            Only used if `nlive` is not explicitly provided in ``**kwargs``.
        **kwargs
            Additional keyword arguments passed directly to ``pypolychord.run``.
            Common arguments include ``nlive``, ``num_repeats``, etc.

        Returns
        -------
        AnestheticResults
            The results object containing the fitted model and the `anesthetic.NestedSamples` object.
        """
        # Dynamic imports
        import pypolychord
        
        if not 'nlive' in kwargs and nlive_factor is not None:
            kwargs['nlive'] = nlive_factor * ctx.num_params

        if ctx.output_path is not None:
            kwargs.setdefault('base_dir', f'{ctx.output_path}/chains')
        kwargs.setdefault('file_root', ctx.output_root)
        
        # Get the model parameters
        param_names = ctx.flat_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'{name_replaced}'] for name, name_replaced in zip(param_names, dot_param_names)])
        
        # Generate prior and likelihood functions
        x0 = np.array(ctx.model.flat_param_values())
        loglikelihood_fn = ctx.make_log_likelihood_fn(as_numpy=True)
        prior_fn = ctx.make_prior_transform_fn(as_numpy=True)
        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: self.logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')

        self.logger.info(f'PolyChord started at {time_string()}')
        nested_samples = pypolychord.run(
            loglikelihood_fn,
            len(param_names),
            dumper=dumper,
            prior=prior_fn,
            paramnames=labeled_param_names,
            **kwargs
        )
        
        self.logger.info(f'PolyChord finished at {time_string()}')
        
        for i, param_name in enumerate(param_names[0:-ctx.num_likelihood_params]):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
                
        fitted_model = ctx.model.with_params(x0)
        
        return AnestheticResults(fitted_model=fitted_model, solver_results=nested_samples)