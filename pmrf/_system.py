from typing import final, Callable, Any

import skrf
import jax
import equinox as eqx

import pmrf.numpy as np
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._tree import nodes_at_paths


class SharedModel:
    """Placeholder value for models that have been removed."""

    def __repr__(self):
        return "SharedModel"

class SystemModel(Model):
    """ A `SystemModel` is a collection of related models, grouped together into a single N-port model with some extra functionality.

    Sometimes, it may be necessary to combine multiple related models into a single, larger model. The most common use-case for this
    is when lower-level models need to be shared amongst several higher-level models. Since models in `paramrf` are designed
    to be functional and effectively immutable, however, regular sharing of object references is not supported.

    Another added functionality is the automatic implementation of the `s` method. This combines the S-parameters of sub-models
    into a large matrix, with coupling  between ports across models equal to zero. This functionality is mainly is intended
    for use with `jax`, where the unnecessary zero columns would be jit-compiled away.
    
    As an example, one might want to model a high-frequency circuit board with a number of different inputs but with some transmission lines
    in the path shared. In that case, in might make sense to model these inputs as separate models that each reference the same underlying
    transmission line model.
    
    `SystemModel` is provided as an easy-to-use solution to cater for the above, with the goal of acting as a high-level abstraction
    that easily allows efficient model sharing. By simply inheriting from `SystemModel`, all `Model` objects that share they same
    name (not `None`) and instance will shared, independent of where they are used in the model. Further, any necessary abstract methods (e.g. `s`, `y`)
    will be conveniently implemented to return large, stacked matrices of the top-level models for ease-of-manipulation, but methods such as `to_skrf`
    are overriden to return the networks individually (by default), as would usually be desired.
    """
    _get: Callable | None = eqx.field(default=None, init=False, repr=False)
    _where: Callable | None = eqx.field(default=None, init=False, repr=False)
    # _shared_locations: tuple = eqx.field(static=True, init=False, repr=False)
    # _treedef: Any = eqx.field(static=True, init=False, repr=False)
    # _path_to_idx: dict = eqx.field(static=True, init=False, repr=False)

    @final
    def post(self):
        # TODO raise error or deal with NESTED sharing
        pass        
        # First, build a map of the unique model identifier (id + name) to all its paths in the models
        # models_path_vals = self.nested_submodels_with_paths
        # id_to_paths = {}
        # for path, model in models_path_vals:
        #     key = (id(model), model.name)
        #     id_to_paths.setdefault(key, [])
        #     id_to_paths[key].append(path)
            
        # # Next, collect the paths in the map that are shared (more than one path per identifier) and choose the first as the "base" and the rest as the "replace".
        # # We repeat the first however many times we must replace it so we have a one-to-one mapping for each replace
        # models_paths = [paths for paths in id_to_paths.values() if len(paths) > 1]
        # models_base_paths = [[model_paths[0]] * len(model_paths[1:]) for model_paths in models_paths]
        # models_replace_paths = [model_paths[1:] for model_paths in models_paths]

        # # Flatten and store the base/replace paths for all shared models. Really ugly because of all the singular/plurals but gets the job done
        # base_paths = [base_path for model_base_paths in models_base_paths for base_path in model_base_paths]
        # replace_paths = [replace_path for model_replace_paths in models_replace_paths for replace_path in model_replace_paths]
        
        # # Generate the get/where functions for the eqx.tree_at, and remove any duplicate nodes
        # self._get = lambda model: nodes_at_paths(model, base_paths)
        # self._where = lambda model: nodes_at_paths(model, replace_paths)
        
        # Finally, removed the shared models in-place. They will be reconstructed at call time for s() etc.
        # self.__dict__.update(self.shared().__dict__)
        
    def shared(self) -> 'SystemModel':
        return self
        return eqx.tree_at(self._where, self, replace_fn=lambda _: SharedModel())
    
    def reconstructed(self) -> 'SystemModel':
        # return eqx.tree_at(self._where, self, self._get(self))
        return self

    def s(self, freq: Frequency) -> np.ndarray:
        nports = 0
        submodels = self.reconstructed().submodels
        for submodel in submodels:
            nports += submodel.nports

        s = np.zeros((freq.npoints, nports, nports))
        i = 0
        for submodel in submodels:
            s_sub = submodel.s(freq)
            for m, n in submodel.port_tuples:
                s = s.at[:,i+m,i+n].set(s_sub[m,n])
            i += submodel.nports**2
        return s
    
    def to_skrf(self, frequency: skrf.Frequency | list[skrf.Frequency], **kwargs) -> list[skrf.Network]:
        models = self.reconstructed().submodels
        networks = []

        if not isinstance(frequency, list):
            frequency = [frequency] * len(models)

        if isinstance(frequency, list):
            for model, model_frequency in zip(models, frequency):
                networks.append(model.to_skrf(model_frequency, **kwargs))
        else:
            model_frequency = frequency
            for model in models:
                networks.append(model.to_skrf(model_frequency, **kwargs))
        return networks