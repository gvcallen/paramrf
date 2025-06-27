import skrf
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db
from pmrf._model import Model
from pmrf._constants import FeatureT, FeatureListT, ArrayFuncT

from pmrf.fitting._base import BaseFitter, FitResults
from pmrf.fitting._features import generate_feature_function

class FrequentistResults(FitResults):
    pass

class FrequentistFitter(BaseFitter):
    """
    **Overview**

    A base class for frequentist (classical) optimization methods.

    This class extends `BaseFitter` by adding the concept of a `cost_fn`,
    a function that takes the difference between model features and measured
    features and computes a single scalar value representing the "cost" or "error".
    The goal of the fitter is to minimize this value.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureT | FeatureListT | None = None,
        cost: ArrayFuncT | list[ArrayFuncT] | eqx.Module = None,
        *args, **kwargs
    ) -> None:
        """Initializes the FrequentistFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. Defaults to `None`.
            features (FeatureT | FeatureListT | None, optional),
                The features to extract for comparison. Defaults to `None`.
            cost (ArrayFuncT | list[ArrayFuncT] | eqx.Module, optional):
                A function or sequence of functions defining the cost metric. If a list
                of functions is provided, they are composed sequentially. If `None`, a
                default cost function (typically L2 norm on the dB magnitude difference)
                is used. Defaults to `None`.
        """
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        if cost is not None and not isinstance(cost, list):
            cost = [cost]
        if cost is None:
            if len(features) > 1:
                cost = [mag_2_db, l2_norm_ax0, l2_norm_ax0]
            else:
                cost = [mag_2_db, l2_norm_ax0]
        self.cost_fn = cost if isinstance(cost, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost])
        
    def _generate_numpy_cost_function(self, dont_jit=False):
        # Generate JAX-compatible functions for feature extraction and model reconstruction
        feature_fn, x0, recon_fn = generate_feature_function(self.model, self.feature_list, self.model_frequency, flat=True)

        # Define the JAX cost function to be minimized
        def cost_jax(flat_params) -> jnp.ndarray:
            model_features = feature_fn(flat_params)
            error = self.measured_features - model_features
            return self.cost_fn(error)

        if not dont_jit:
            self.logger.info("Compiling cost function...")
            cost_jax = jax.jit(cost_jax)
        
        def cost_numpy(x):
            flat_params_jax = jnp.array(x)
            return float(cost_jax(flat_params_jax))
            
        return cost_numpy, recon_fn, x0

class ScipyMinimizeFitter(FrequentistFitter):
    """
    **Overview**

    A concrete fitter that uses the powerful `scipy.optimize.minimize` function
    to find the optimal model parameters.

    This class provides a versatile interface to the wide variety of gradient-based
    and gradient-free optimization algorithms available in SciPy, making it a
    robust choice for many fitting problems. The cost function is internally JIT-compiled
    with JAX for performance.

    **Example**

    ```python
    import pmrf as prf
    import skrf as rf

    # 1. Load measurement data
    measured_ntwk = rf.Network('my_device.s2p')

    # 2. Create the parametric model to fit
    model_to_fit = prf.models.DatasheetCoaxial(length=0.1)

    # 3. Instantiate and configure the fitter
    fitter = prf.fitting.ScipyMinimizeFitter(
        model=model_to_fit,
        measured=measured_ntwk,
        features=[('s_db', (0,0)), ('s_db', (1,0))] # Fit to S11 and S21 in dB
    )

    # 4. Run the fit, passing optimizer-specific options via kwargs
    # Here, we use the 'SLSQP' algorithm.
    fit_result = fitter.run(method='SLSQP')

    # 5. Print the results
    print("Fit complete. Optimized model:")
    print(fit_result.model)
    ```
    """
    def run(self, *args, **kwargs) -> FrequentistResults:
        """Executes the optimization using `scipy.optimize.minimize`.

        This method sets up the JAX-compiled cost function, defines parameter
        bounds, and invokes the SciPy minimizer.

        Args:
            *args: Positional arguments passed directly to `scipy.optimize.minimize`.
            **kwargs: Keyword arguments passed directly to `scipy.optimize.minimize`,
                      allowing full control over the solver (e.g., `method='BFGS'`,
                      `tol=1e-6`).

        Returns:
            FrequentistResults: An object containing the optimized model and results.
        """
        from scipy.optimize import minimize, Bounds
        
        # Extract parameter values and bounds from the model
        params_list = list(self.model.to_params_list())
        minimums = [p.min for p in params_list]
        maximums = [p.max for p in params_list]
        bounds = Bounds(minimums, maximums)

        cost_fn, recon_fn, x0 = self._generate_numpy_cost_function()
        
        # Define a wrapper function compatible with SciPy's interface
        def cost_wrapper(x, callback_args):
            cost = cost_fn(x)
            i = callback_args['fevel']
            if i % 500 == 0:
                self.logger.info(f"fevel = {i}, cost = {cost:.2f}")
            callback_args['fevel'] += 1
            return cost

        callback_args = {'fevel': 0}
        scipy_result = minimize(cost_wrapper, x0, args=(callback_args,), bounds=bounds, *args, **kwargs)
        self.logger.info(f"Optimization finished: {scipy_result.message}")
        
        # Reconstruct the final model with optimized parameters
        model_opt = recon_fn(scipy_result.x)
        return FrequentistResults(model=model_opt, engine_results=scipy_result)