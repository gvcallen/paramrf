import skrf

from pmrf._compound import CompoundModel
from pmrf._model import Model

class SystemModel(CompoundModel):
    """ A `SystemModel` is a collection of related models.

    Sometimes, it is necessary to combine multiple related models into a single, larger model. The most common use-case for this
    is when lower-level models need to be shared amongst several higher-level models. However, since models in `paramrf`
    are designed to be functional (and effectively immutable), regular object-orientated forward/backward references are discouraged.
    
    As an example, one might want to model a high-frequency circuit board with a number of different inputs but with some transmission lines
    in the path shared. In that case, in might make sense to model these inputs as separate models that each reference the same underlying
    transmission line model.
    
    `SystemModel` is provided as an easy-to-use solution to cater for the above, with the goal of acting as a high-level abstraction
    that easily allows efficient model sharing. By simply inheriting from `SystemModel`, all `Model` objects that share they same
    name will shared, independent of where they are used in the model. Further, any necessary abstract methods (e.g. `s`, `y`)
    will be conveniently implemented to return large, stacked matrices of the top-level models, but methods such as `to_skrf`
    are overriden to return the networks individually by default, as is usually desired.
    """
    @property
    def models(self) -> list[Model]:
        raise NotImplementedError("'models' not yet implemented for SystemModel")

        
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