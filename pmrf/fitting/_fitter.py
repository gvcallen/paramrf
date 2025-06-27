from abc import ABC, abstractmethod

import logging
import skrf
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._constants import FeatureListT, ArrayFuncT
from pmrf._tree import combine

from pmrf.fitting._features import extract_features, generate_feature_function
from pmrf.fitting._results import FitResults, FrequentistResults, BayesianResults

class BaseFitter(ABC):
    """
    **Overview**

    An abstract base class that provides the foundational structure for all
    fitting algorithms in `pmrf`.

    This class handles the common setup tasks required for any fitting routine,
    including:
    - Managing the parametric `Model` to be optimized.
    - Processing and aligning the measured `skrf.Network` data.
    - Interpolating all data onto a common frequency axis.
    - Defining the logic for feature extraction, which transforms raw S-parameters
      into a format suitable for comparison (e.g., magnitude, dB, phase).
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureListT = None,
    ) -> None:
        """Initializes the BaseFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against. If a list of
                networks is passed, they are treated as a single stacked N-port network.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. If `None`, the frequency
                from the first measured network is used. All networks will be
                interpolated onto this single frequency axis. Defaults to `None`.
            features (FeatureListT, optional):
                Defines the features to be extracted from the network data for comparison.
                This can be a list of strings (e.g., `['s_db', 's_deg']`) to extract
                those features for all ports, or a list of (feature, ports) tuples
                (e.g., `[('s_db', (0,0)), ('s_db', (1,1))]`) for more specific extraction.
                Defaults to `None`, which uses S11 magnitude (`('s', (0, 0))`).
        """
        features = features or [('s', (0, 0))]
        
        # All frequencies must be the same across all measurements (at least currently..)
        measured = [measured] if not isinstance(measured, list) else measured
        if frequency is not None:
            measured = [ntwk.interpolate(frequency) for ntwk in measured]
            measured_freq = frequency
        else:
            measured_freq = measured[0].frequency
            for ntwk in measured:
                if ntwk.frequency != measured_freq and not len(ntwk.frequency) == 0:
                    raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency for fitting)")
                
        # Initialize model parameters from user and store in flat array
        self.model: Model = model
        self.model_frequency = Frequency.from_skrf(measured_freq)
        self.measured: list[skrf.Network] = measured
        self.measured_frequency = measured_freq
        self.measured_features = extract_features(measured, features)
        self.feature_list = features
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def run(self, *args, **kwargs) -> FitResults:
        """Executes the fitting algorithm.

        This method must be implemented by all concrete subclasses. It is the
        main entry point to start the optimization or sampling process.

        Returns:
            FitResults: An object containing the results of the fit.
        """
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
        features: FeatureListT | None = None,
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
            features (FeatureListT, optional):
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
        params_list = list(self.model.params.values())
        minimums = [p.min for p in params_list]
        maximums = [p.max for p in params_list]
        bounds = Bounds(minimums, maximums)

        # Generate JAX-compatible functions for feature extraction and model reconstruction
        feature_fn, x0, recon_fn = generate_feature_function(
            self.model, self.feature_list, self.model_frequency, flat=True
        )

        # Define the JAX cost function to be minimized
        def cost_jax(flat_params) -> jnp.ndarray:
            model_features = feature_fn(flat_params)
            error = self.measured_features - model_features
            return self.cost_fn(error)

        self.logger.info("Compiling cost function with JAX...")
        cost_jit = jax.jit(cost_jax)
        
        # Define a wrapper function compatible with SciPy's interface
        def cost_scipy(x, callback_args):
            flat_params_jax = jnp.array(x)
            cost = float(cost_jit(flat_params_jax))
            
            i = callback_args['fevel']
            if i % 500 == 0:
                self.logger.info(f"fevel = {i}, cost = {cost:.2f}")
            callback_args['fevel'] += 1
            return cost

        callback_args = {'fevel': 0}
        self.logger.info("Starting SciPy optimization...")
        scipy_result = minimize(cost_scipy, x0, args=(callback_args,), bounds=bounds, *args, **kwargs)
        self.logger.info(f"Optimization finished: {scipy_result.message}")
        
        # Reconstruct the final model with optimized parameters
        model_opt = recon_fn(scipy_result.x)
        return FrequentistResults(model=model_opt, engine_results=scipy_result)