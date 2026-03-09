from functools import partial
from abc import ABC
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.fitting.base import BaseFitter
from pmrf.models.model import Model
from pmrf.constants import ArrayFuncT

def independent_rms_error(res):
    feature_rms = jnp.sqrt(jnp.mean(jnp.abs(res)**2, axis=0))
    combined_rms = jnp.sqrt(jnp.mean(feature_rms**2))
    return 20*jnp.log10(combined_rms)

def convolutional_rms_error(res, num_features=2):
    feature_rms = jnp.sqrt(jnp.mean(jnp.abs(res)**2, axis=0))
    
    grouped = feature_rms.reshape(-1, num_features)
    M = grouped.shape[0]
    convolved = grouped[:, 0]
    for i in range(1, num_features):
        convolved = jnp.convolve(convolved, grouped[:, i])
        
    convolved_scaled = convolved / (M ** (num_features - 1))
    rms_convolved = jnp.sqrt(jnp.mean(convolved_scaled**2))
    combined_rms = rms_convolved ** (1.0 / num_features)
    
    return 20 * jnp.log10(combined_rms)

def geometric_rms_error(res, num_features=2):
    feature_rms = jnp.sqrt(jnp.mean(jnp.abs(res)**2, axis=0))
    
    grouped = feature_rms.reshape(-1, num_features)
    global_rms = jnp.sqrt(jnp.mean(grouped**2, axis=0))
    product = jnp.prod(global_rms)
    combined_rms = product ** (1.0 / num_features)
    
    return 20 * jnp.log10(combined_rms)

class FrequentistFitter(BaseFitter, ABC):
    r"""
    A base class for frequentist (classical) optimization methods.
    
    This class sets up the objective function (or cost function) needed by standard 
    numerical optimizers (like those in SciPy). It automatically configures the 
    target features and error metrics based on the type of fit you want to perform.

    .. rubric:: Methods

    .. autosummary::
       :nosignatures:
       
       run
       execute
       cost

    Parameters
    ----------
    model : Model
        The ParamRF model containing the parameters to optimize.
    cost_kind : str, optional
        A preset string to automatically configure both the features and error function
        into a combined cost function. Options are 'complex', 'magnitude', 'geometric', or 'convolutional'.
        'convolutional' and 'geometric' use custom error functions with the same names, whereas
        'complex' and 'magnitude' use the 'independent' error function. If left None, and no features
        or error function is provided, then 'complex' is assumed.
    error_kind : str, optional:
        A preset string to automatically configure an error function. Options are 'independent',
        'geometric' or 'convolutional'. Note that all of these currently use the L2 norm (RMS).
    error_fn : callable, optional
        The specific function used to calculate the error. Passed the difference
        between the model and measured features. If not provided, is configured based on `cost_kind`
        and `error_kind`.
    tikhonov_lambda : float, default=0.0
        The weight to use for Tikhonov regularization. The weight is multiplied to a regularization
        term which depends on the squared difference between the new model parameters and
        the initial model parameters in normalized space.
    **kwargs
        Additional arguments passed up to :class:`BaseFitter` (such as ``frequency``).
    """
    def __init__(
        self,
        model: Model, 
        *,
        cost_kind: str = None,
        error_kind: str = None,
        error_fn: ArrayFuncT | None = None,
        tikhonov_lambda: float = 0.0,
        **kwargs
    ):
        # Let base class consume features, etc.
        super().__init__(model, **kwargs)

        self.tikhonov_lambda = tikhonov_lambda
        self.initial_theta = self.model.flat_param_values()

        # Extract bounds and calculate safe parameter ranges
        lower_bounds, upper_bounds = self.model.distribution().bounds
        param_ranges = jnp.array(upper_bounds) - jnp.array(lower_bounds)
        self.param_ranges = jnp.where(param_ranges == 0.0, 1.0, param_ranges)        
        
        # Apply standard Python defaults/mutations safely to 'self'
        if cost_kind is None and (self.features is None and error_fn is None):
            cost_kind = 'complex'
        elif cost_kind is not None and (error_fn is not None or self.features is not None):
            raise ValueError("Cannot pass a cost kind alias with an error function or specific features")
        
        if cost_kind == 'convolutional':
            self.features = ['s', 's_mag']
            error_kind = 'convolutional'
        elif cost_kind == 'geometric':
            self.features = ['s', 's_mag']
            error_kind = 'geometric'
        elif cost_kind == 'complex':
            self.features = ['s']
            error_kind = 'independent'
        elif cost_kind == 'magnitude':
            self.features = ['s_mag']
            error_kind = 'independent'
        else:
            raise Exception("Unknown cost kind")
                
        if error_kind is None and error_fn is None:
            error_kind = 'independent'
        elif error_kind is not None and error_fn is not None:
            raise Exception("Cannot pass an error function with an error kind or cost kind alias")
        
        if error_fn is None:
            if error_kind == 'independent':
                error_fn = independent_rms_error
            elif error_kind == 'geometric':
                error_fn = partial(geometric_rms_error, num_features=len(self.features))
            elif error_kind == 'convolutional':
                error_fn = partial(convolutional_rms_error, num_features=len(self.features))
            else:
                raise Exception("Unknown error kind")
            
        self._error_fn = eqx.filter_jit(error_fn)
        self._cost_fn = None

    def cost(self, theta: jnp.ndarray, target_features: jnp.ndarray) -> jnp.ndarray:
        r"""
        Calculate the scalar error (cost) for a given set of parameters.
        
        This function computes the model features, calculates the residual against 
        the target data, and passes it through the defined error function. The 
        entire calculation is lazily compiled using ``jax.jit`` on the first call 
        to ensure the optimization loop runs as fast as possible.
        
        Parameters
        ----------
        theta : jax.numpy.ndarray
            A 1D array containing the current parameter values being tested by 
            the optimizer.
        target_features : jax.numpy.ndarray
            The extracted measurement data that the optimizer is trying to match.

        Returns
        -------
        jax.numpy.ndarray
            The scalar cost value representing the total error.
        """
        if self._cost_fn is None:
            self.logger.debug("Lazily compiling Frequentist cost function...")

            # Capture class attributes for the JAX closure at compile time
            reg_weight = self.tikhonov_lambda
            theta_0 = self.initial_theta
            p_ranges = self.param_ranges
            
            # 2. Compile the specific cost loop
            @jax.jit
            def cost_fn(theta, target_feats):
                model_features = self.model_features(theta)
                error = target_feats - model_features

                # Base cost
                base_cost = self._error_fn(error)
                base_cost = base_cost if jnp.isscalar(base_cost) else base_cost[0]

                if reg_weight > 0.0 and theta_0 is not None:
                    # Normalize the difference by the parameter ranges
                    normalized_diff = (theta - theta_0) / p_ranges
                    l2_penalty = reg_weight * jnp.sum(normalized_diff ** 2)
                    return base_cost + l2_penalty                

                cost_val = self._error_fn(error)
                return cost_val if jnp.isscalar(cost_val) else cost_val[0]
                
            self._cost_fn = cost_fn
            
        # 3. Evaluate the cached function
        return self._cost_fn(jnp.array(theta), jnp.array(target_features))