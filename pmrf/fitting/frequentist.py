import numpy as np
import skrf
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db, conv_inter
from pmrf.models.model import Model
from pmrf.constants import FeatureInputT, ArrayFuncT
from pmrf.fitting.base import BaseFitter, FitResults

L2_COST = [l2_norm_ax0, l2_norm_ax0, mag_2_db]
CONVOLUTIONAL_COST = [l2_norm_ax0, conv_inter, l2_norm_ax0, mag_2_db]

class FrequentistResults(FitResults):
    pass

class FrequentistFitter(BaseFitter):
    """
    A base class for frequentist (classical) optimization methods.

    This class extends `BaseFitter` by adding the concept of a `cost_fn`,
    a function that takes the difference between model features and measured
    features and computes a single scalar value representing the "cost" or "error".
    """
    def __init__(
        self,
        model: Model,
        *,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        output_root: str = 'fit',
        sparam_kind: str = 'all',        
        cost_kind: str | None = None,
        cost_function: ArrayFuncT | list[ArrayFuncT] | eqx.Module = None,
        **kwargs
    ) -> None:
        """Initializes the FrequentistFitter.

        Args:
            model (Model):
                The parametric `pmrf` Model to be fitted.
            features (FeatureInputT | None, optional):
                Defines the features to be extracted from the model and network(s).
                Defaults to `None`, in which case real and imaginary features for all ports are used.
                Can be a single feature e.g. 's11', a list of features (e.g., `['s11', 's11_mag']`),
                or a dictionary with either of the above as value. In the dictionary case,
                keys must be network names in the collection passed by `measured`, which must also
                correspond to submodels which are attributes of the model. For example,
                {'name1', ('s11'), {'name2', ('s21')} can be passed.
                Note that if a collection of networks is passed, but a feature dictionary is not,
                it is assumed that those feature(s) should be extract for each networks/submodel.
                See `extract_features(..)` more details.
            output_path (str | None):
                The path for fitters to write output data to. Defaults to `None`.
            output_root (str | None):
                The root name used for output files in the output path. Defaults to `None`.
            sparam_kind (str | None):
                The S-parameter data kind to use for port-expansion in feature extraction. Can either be 'transmission', 'reflection' or 'all'.
                See `extract_features` for more details.              
            cost_kind (str, optional):
                A cost 'kind' alias to initialize the feature extractors and cost function from.
                Can be one of 'convolutional', 'complex', or 'magnitude'.
            cost_function (ArrayFuncT | list[ArrayFuncT] | eqx.Module, optional):
                A function or sequence of functions defining the cost metric. If a list
                of functions is provided, they are composed sequentially. If `None`,
                then `cost_kind` defines the cost function. Defaults to `None`.
        """
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
        else:
            raise Exception("Unknown cost kind alias passed to frequentist fitter")

        if features is None:
            features = default_features
        if cost_function is None:
            cost_function = default_cost
        
        super().__init__(model=model, features=features, output_path=output_path, output_root=output_root, sparam_kind=sparam_kind, **kwargs)

        features = self._active_feature_spec
        if cost_function is not None and not isinstance(cost_function, list):
            cost_function = [cost_function]
        if cost_function is None:
            if len(features) > 1:
                cost_function = [l2_norm_ax0, l2_norm_ax0, mag_2_db]
            else:
                cost_function = [l2_norm_ax0, mag_2_db]
        
        self.cost_metric_fn = cost_function if isinstance(cost_function, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost_function])

        
    def _make_cost_function(self, as_numpy=False):
        x0_jax = self._active_model.flat_params()
        feature_fn_jax = self._make_feature_function()

        # Define the JAX cost function to be minimized
        @jax.jit
        def cost_fn(flat_params) -> jnp.ndarray:
            model_features = feature_fn_jax(flat_params)
            error = self._active_measured_features - model_features
            cost_val = self.cost_metric_fn(error)
            if jnp.isscalar(self.cost_metric_fn(error)):
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
    
    def _bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        param_groups = self._active_model.param_groups()
        param_names = self._active_model.flat_param_names()
        
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