from abc import abstractmethod

import skrf
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
    name will shared, independent of where they are used in the model. Further, any necessary abstract methods (e.g. `s`, `y`)
    will be conveniently implemented to return large, stacked matrices of the top-level models for ease-of-manipulation, but methods such as `to_skrf`
    are overriden to return the networks individually (by default), as would usually be desired.
    """
    # @abstractmethod
    # def build(self):
    #     raise NotImplementedError("Error: system model sub-classes *have* to implement the build() function to build their sub-models")

    def s(self, freq: Frequency) -> np.ndarray:
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
