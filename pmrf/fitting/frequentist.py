from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db, conv_inter
from pmrf.models.model import Model
from pmrf.constants import ArrayFuncT
from pmrf.fitting.base import BaseFitter, FitResults, FitContext

L2_COST = [l2_norm_ax0, l2_norm_ax0, mag_2_db]
CONVOLUTIONAL_COST = [l2_norm_ax0, conv_inter, l2_norm_ax0, mag_2_db]

class FrequentistResults(FitResults):
    """
    Results obtained from a Frequentist (classical) fitting process.
    
    This class is a marker subclass of `FitResults` and currently shares the same structure.
    """
    pass

@dataclass
class FrequentistContext(FitContext):
    """
    Context object for Frequentist fitting, containing the cost function.

    Attributes
    ----------
    cost_function : eqx.Module or None
        The sequence of functions defining the error metric.
    """
    cost_function: eqx.Module | None = None
    
    def make_cost_function(self, as_numpy=False):
        """
        Create the cost function to be minimized.

        The cost function calculates the error between measured and model features,
        applies the defined `cost_function` transformation, and returns a scalar value.

        Parameters
        ----------
        as_numpy : bool, optional, default=False
            If True, returns a function compatible with NumPy arrays; otherwise JAX arrays.

        Returns
        -------
        callable
            The JIT-compiled cost function taking flat parameters and returning a scalar cost.
        """
        x0_jax = self.model.flat_param_values()
        feature_fn_jax = self.make_feature_function()

        # Define the JAX cost function to be minimized
        @jax.jit
        def cost_fn(flat_params) -> jnp.ndarray:
            model_features = feature_fn_jax(flat_params)
            error = self.measured_features - model_features
            cost_val = self.cost_function(error)
            if jnp.isscalar(self.cost_function(error)):
                return cost_val
            else:
                return cost_val[0]
            
        if as_numpy:
            cost_fn_jax = cost_fn
            cost_fn = lambda x: float(cost_fn_jax(jnp.array(x)))
            x0_np = np.array(x0_jax)
            x0 = x0_np
            
        self.logger.info(f"Compiling cost function...")
        _cost_val = cost_fn(x0)
        return cost_fn
    
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Retrieve the lower and upper bounds for all model parameters.

        Returns
        -------
        tuple of jnp.ndarray
            A tuple containing (lower_bounds, upper_bounds).

        Raises
        ------
        Exception
            If any parameter is not associated with a parameter group.
        """
        param_groups = self.model.param_groups()
        param_names = self.model.flat_param_names()
        
        name_to_minimum = {name: None for name in param_names}
        name_to_maximum = {name: None for name in param_names}
        
        for param_group in param_groups:
            group_minimums, group_maximums = param_group.min, param_group.max
            group_param_names = param_group.parameter_names
            for i, name in enumerate(group_param_names):
                name_to_minimum[name] = group_minimums[i]
                name_to_maximum[name] = group_maximums[i]
        
        if any(value is None for value in name_to_minimum.values()) or any(value is None for value in name_to_maximum.values()):
            raise Exception('Parameter found that did not belong to a parameter groups')
        
        return jnp.array(list(name_to_minimum.values())), jnp.array(list(name_to_maximum.values()))    

class FrequentistFitter(BaseFitter):
    """
    A base class for frequentist (classical) optimization methods.

    This class extends `BaseFitter` by adding the concept of a cost function,
    which takes the difference between model features and measured
    features and computes a single scalar value representing the "cost" or "error".
    """
    def __init__(
        self,
        model: Model,
        *,
        cost_kind: str | None = None,
        cost_function: ArrayFuncT | list[ArrayFuncT] | eqx.Module | None = None,
        **kwargs
    ) -> None:
        """
        Initializes the FrequentistFitter.

        Parameters
        ----------
        model : Model
            The parametric `pmrf` Model to be fitted.
        cost_kind : str, optional
            A cost 'kind' alias to initialize the features and cost function from.
            Can be one of 'convolutional', 'complex', or 'magnitude'.
        cost_function : ArrayFuncT, list[ArrayFuncT] or eqx.Module, optional
            A function or sequence of functions defining the cost metric. If a list
            of functions is provided, they are composed sequentially. If `None`,
            then `cost_kind` defines the cost function. Defaults to `None`.
        **kwargs
            Additional arguments forwarded to :class:`BaseFitter`.
        """
        self.cost_kind = cost_kind
        self.cost_function = cost_function
                            
        super().__init__(model, **kwargs)

    def create_context(self, measured, *, cost_kind=None, cost_function=None, **kwargs) -> FrequentistContext:
        """
        Create a FrequentistContext for the fitting process.

        Parameters
        ----------
        measured : skrf.Network or NetworkCollection
            The measured data.
        cost_kind : str, optional
            Override for the cost kind alias.
        cost_function : callable or list of callables, optional
            Override for the cost function.
        **kwargs
            Additional context arguments (e.g., `features`).

        Returns
        -------
        FrequentistContext
            The configured Frequentist context.
        
        Raises
        ------
        Exception
            If an unknown `cost_kind` alias is provided.
        """
        features = kwargs.pop('features', None) or self.features
        cost_kind = cost_kind or self.cost_kind
        cost_function = cost_function or self.cost_function
        
        default_features = None
        default_cost = None
        if cost_kind == 'convolutional':
            default_features = ['s', 's_mag']
            default_cost = CONVOLUTIONAL_COST
        elif cost_kind == 'complex':
            default_features = ['s']
            default_cost = L2_COST
        elif cost_kind == 'magnitude':
            default_features = ['s_mag']
            default_cost = L2_COST
        elif cost_kind is not None:
            raise Exception("Unknown cost kind alias passed to frequentist fitter")

        if features is None:
            features = default_features
        if cost_function is None:
            cost_function = default_cost        
        
        base_ctx = super().create_context(measured, features=features, **kwargs)
    
        feature_spec = base_ctx.features
        if cost_function is not None and not isinstance(cost_function, list):
            cost_function = [cost_function]
        if cost_function is None:
            if len(feature_spec) > 1:
                cost_function = [l2_norm_ax0, l2_norm_ax0, mag_2_db]
            else:
                cost_function = [l2_norm_ax0, mag_2_db]
        
        cost_function = cost_function if isinstance(cost_function, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost_function])
        
        return FrequentistContext(
            model=base_ctx.model,
            measured=base_ctx.measured,
            frequency=base_ctx.frequency,
            features=base_ctx.features,
            measured_features=base_ctx.measured_features,
            output_path=base_ctx.output_path,
            output_root=base_ctx.output_root,
            sparam_kind=base_ctx.sparam_kind,
            cost_function=cost_function,
            logger=self.logger,
        )