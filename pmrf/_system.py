from typing import final, Callable, Any

import skrf
import jax
import equinox as eqx

import pmrf.numpy as np
from pmrf._model import Model
from pmrf._frequency import Frequency

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
    _shared_locations: tuple = eqx.field(static=True, init=False, repr=False)
    _treedef: Any = eqx.field(static=True, init=False, repr=False)
    _path_to_idx: dict = eqx.field(static=True, init=False, repr=False)

    @final
    def post(self):
        # TODO allow more than one-level sharing
        model_path_vals = self.submodels_with_paths
        
        # 2. Group the found modules by their Python `id()` to find shared instances.
        id_to_paths = {}
        for path, mod in model_path_vals:
            id_to_paths.setdefault(id(mod), []).append(path)    

        # These are groups of paths, where each group points to the same module object.
        shared_module_groups = [
            tuple(paths) for paths in id_to_paths.values() if len(paths) > 1
        ]

        # 3. For each group of shared modules, find the paths to their corresponding leaves.
        # The substitution logic operates on leaves, so we pre-calculate their locations.
        final_shared_leaf_locations = []
        if shared_module_groups:
            # Define a helper to retrieve a module from the model using its path.
            def get_node(tree, path):
                for key in path:
                    tree = key.from_node(tree)
                return tree

            for module_path_group in shared_module_groups:
                # Use the first module in the group as the canonical reference.
                canonical_module_path = module_path_group[0]
                canonical_module = get_node(self, canonical_module_path)
                
                # Flatten the canonical module to find its internal leaf structure.
                module_leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(canonical_module)

                # For each leaf inside the module, create a group of its full paths
                # across all shared module instances.
                for relative_leaf_path, _ in module_leaves_with_paths:
                    leaf_path_group = [
                        module_path + relative_leaf_path for module_path in module_path_group
                    ]
                    final_shared_leaf_locations.append(tuple(leaf_path_group))
        
        # 4. Flatten the entire model to get the definitions needed for reconstruction.
        paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(self)
        
        # 5. Store the results on the now-frozen Equinox module instance.
        object.__setattr__(self, '_shared_locations', tuple(final_shared_leaf_locations))
        object.__setattr__(self, '_treedef', treedef)
        object.__setattr__(self, '_path_to_idx', {path: i for i, (path, _) in enumerate(paths_and_leaves)})

    def s(self, freq: Frequency) -> np.ndarray:
        # If discovery didn't run or found no shared nodes, just call the original
        if not hasattr(self, '_shared_locations') or not self._shared_locations:
            return self._s(freq)

        # 1. Get the up-to-date leaves from the current model state
        leaves = jax.tree_util.tree_leaves(self)
        mutable_leaves = list(leaves)
        
        # 2. Substitute shared values
        for group in self._shared_locations:
            canonical_path = group[0]
            canonical_idx = self._path_to_idx[canonical_path]
            canonical_value = mutable_leaves[canonical_idx]
            for path in group[1:]:
                idx_to_update = self._path_to_idx[path]
                mutable_leaves[idx_to_update] = canonical_value

        # 3. Reconstruct the model with the corrected leaves and call _s
        recon_self = self._treedef.unflatten(mutable_leaves)
        return recon_self._s(freq)
    
    def _s(self, freq: Frequency) -> np.ndarray:
        nports = 0
        submodels = self.submodels
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
        models = self.submodels
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