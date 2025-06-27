import jax
import jax.numpy as jnp
import skrf

from pmrf.parameters import Parameter
from pmrf._model import Model
from pmrf._constants import FeatureT, FeatureListT
from pmrf.fitting._base import BaseFitter, FitResults
from pmrf.fitting._features import make_feature_function
from pmrf._util import time_string

# from numpyro.distributions import Normal, Distribution
# def gaussian_log_likelihood(y_meas, y_pred, sigma):
    # return -0.5 * jnp.sum(((y_meas - y_pred) / sigma)**2 + jnp.log(2 * jnp.pi * sigma**2))

def norm_logpdf(x, loc=0.0, scale=1.0):
    return -0.5 * jnp.log(2 * jnp.pi * scale**2) - 0.5 * ((x - loc)**2) / (scale**2)

def gaussian_log_likelihood(y_meas, y_model, sigma):
    return jnp.sum(norm_logpdf(y_meas, y_model, sigma))

class BayesianResults(FitResults):
    pass
        
class BayesianFitter(BaseFitter):
    """
    **Overview**

    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a likelihood function,
    as well as providing support for prior sampling.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureT | FeatureListT | None = None,
        likelihood: str | None = "gaussian",
        likelihood_params = dict[str, Parameter],
        *args, **kwargs
    ) -> None:
        """Initializes the BayesianFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. Defaults to `None`.
            features (FeatureT | FeatureListT | None = None, optional):
                The features to extract for comparison. Defaults to `None`.
        """
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        self.likelihood_params = likelihood_params
             
class PolychordFitter(BayesianFitter):
    def fit(self, *args, **kwargs):
        # Dynamic imports
        import numpy as np
        import pypolychord
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        
        # Generate prior and likelihood functions
        feature_fn, _x0, _recon_fn = make_feature_function(self.model, self.feature_list, self.model_frequency, flat=True)
        def jax_likelihood(flat_params_with_sigma) -> jnp.ndarray:
            sigma = flat_params_with_sigma[-1]
            model_features = feature_fn(flat_params_with_sigma[0:-1])
            return gaussian_log_likelihood(self.measured_features, model_features, sigma)

        self.logger.info("Compiling model and likelihood function...")
        jax_likelihood = jax.jit(jax_likelihood)
        likelihood_fn = lambda x: float(jax_likelihood(jnp.array(x)))
        priors = [prior for prior in self.model.to_params_list()] + [self.likelihood_params['sigma'].prior]
        prior_fn = lambda hypercube: [prior.icdf(hypercube[i]) for i, prior in enumerate(priors)]
        
        # Get param names
        param_names = self.model.param_names + [param.name for param in self.likelihood_params.values()]
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, dot_param_names)])

        # Run polychord. Useful parameters to investigate may be "precision_criterion" and "synchronous"
        kwargs.update({
            'prior': prior_fn,
            'paramnames': labeled_param_names,
        })
        
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s)...')
        self.logger.info(f'PolyChord thread #{rank} started at {time_string()}')
        
        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: self.logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')
        fit_output = pypolychord.run(
            likelihood_fn,
            len(param_names),
            dumper=dumper,
            **kwargs
        )

        return fit_output