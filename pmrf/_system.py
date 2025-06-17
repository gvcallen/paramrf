import logging
from typing import Callable, Optional, Any, Union, Dict

import skrf

from pmrf._model import Model
from pmrf._compound import CompoundModel
from pmrf._frequency import Frequency
import pmrf.numpy as np
from pmrf.numpy import USE_JAX
from pmrf._pytree import tree_with_params, tree_params

import equinox as eqx
import jax

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
# Add a method to the logger for convenience
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)
logging.Logger.verbose = verbose

logger = logging.getLogger(__name__)


class SystemModel(CompoundModel):
    """ A `SystemModel` is a collection of related models.

    Sometimes, it is necessary to combine multiple related models into a single, larger model. The most common use-case for this
    is when lower-level models need to be shared amongst several higher-level models. However, since models themselves are designed
    to be functional and effectively immutable in `paramrf`, regular object-orientated forward/backward references are discouraged.
    `SystemModel` is therefore provided, with the goal of acting as a high-level abstraction that easily allows efficient model sharing.
    
    As an example, one might want to model a high-frequency circuit board with a number of different inputs but with some transmission lines
    in the path shared. In that case, in might make sense to model these inputs as separate models that each reference the same underlying
    transmission line model.

    The `SystemModel` overrides some of the default model methods with those more tailored towards shared models,
    making it a useful abstraction for general purposes.
    """
    # @property
    # def models(self) -> list[Model]:
    #     raise NotImplementedError("'models' property must be implemented sub-classes for a CompoundModel")

    def with_params(
        self,
        flat_params: Optional[jax.Array] = None,
        separator: str | None = '_',
        submodel_separator: str | None = None,
        array_separator: str | None = None,
        index_separator: str | None = None,
        param_filter: Callable[[Any], bool] = None,
        **params: Any
    ) -> "SystemModel":
        """
        Returns a model system with the specified parameter values.

        This method supports two calling styles:
        1. By keyword: `model.with_params(R=50.0, C=1e-9)`
        2. By flat array: `model.with_params(np.array([50.0, 1e-9]))`

        Args:
            flat_params: A 1D JAX array containing all dynamic parameter 
                         values in their flattened tree order.
            **params: Keyword arguments where keys are the names of the
                       parameters to update and values are their new values.

        Returns:
            A new `ModelSystem` instance with the specified parameters updated for all sub-models.
        """
        return tree_with_params(self, flat_params=flat_params, separator=separator, subtree_separator=submodel_separator, array_separator=array_separator, index_separator=index_separator, param_filter=param_filter, **params)
    
    def params(
        self,
        flat: bool = False,
        separator: str | None = '_',
        submodel_separator: str | None = None,
        array_separator: str | None = None,
        index_separator: str | None = None,
        param_filter: Callable[[Any], bool] | None = None,
    ) -> Union[Dict[str, Any], jax.Array]:
        """Returns a dictionary of human-readable string paths and values for every
        scalar value in the model's flattened parameters.

        This is useful for mapping parameter names to values for external
        solvers, setting bounds, or interpreting results.

        Args:
            separator (str | None, optional): The separator to use for all dividers that are not passed. Defaults to '_'.
            submodel_separator (str | None, optional): The separate before submodels. Defaults to `None`, in which case `separator` is used.
            array_separator (str | None, optional): The separate before array-like parameter. Defaults to `None`, in which case `separator` is used.
            index_separator (str | None, optional): The separator between array sub-indices_. Defaults to `None`, in which case `separator` is used.
            param_filter (Callable[[Any], bool], optional): A filter to determine which fields are considered parameters. Defaults to `None`, in which case only the default `Scalar` and `Vector` types are considered.

        Returns:
            A dictionary of parameter names/paths and values e.g. {'R': 0.0, 'sub_L': 1.0, 'sub.C[0,0]': 2.0, 'sub.C[0,1]': 3.0, ...].
        """ 
        return tree_params(self, flat=flat, separator=separator, subtree_separator=submodel_separator, array_separator=array_separator, index_separator=index_separator, param_filter=param_filter)    
        
    def to_skrf(self, frequency: skrf.Frequency | list[skrf.Frequency], **kwargs) -> list[skrf.Network]:
        networks = []

        if not isinstance(frequency, list):
            frequency = [frequency] * len(self.models)

        if isinstance(frequency, list):
            for model, model_frequency in zip(self.models, frequency):
                networks.append(model.to_skrf(model_frequency, **kwargs))
        else:
            model_frequency = frequency
            for model in self.models:
                networks.append(model.to_skrf(model_frequency, **kwargs))
        return networks