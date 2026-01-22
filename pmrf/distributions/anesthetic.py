from typing import BinaryIO

from pmrf.distributions.serializable import SerializableDistribution

class AnestheticDistribution(SerializableDistribution):
    """
    Adapter for distributions represented by samples from the Anesthetic library.
    
    Although you cannot sample from the nested samples as if they were an actual distribution,
    this class is still useful in storing sampling results directly within a model.
    """
    def __init__(self, nested_samples, param_names: list[str] | None = None, validate_args=None):
        from anesthetic import NestedSamples
        
        if param_names:
            nested_samples = nested_samples.loc[:, param_names]
        else:
            columns = nested_samples.columns
            param_names = [columns[i][0] for i in range(len(columns))]
            param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]
        
        self.nested_samples: NestedSamples = nested_samples
        
        event_shape = (len(param_names),)
        super().__init__(batch_shape=(), event_shape=event_shape, validate_args=validate_args)
        
    def save(self, target: str | BinaryIO):
        if isinstance(target, str):
            return self.nested_samples.to_csv(target)
        return self.write(target)
    
    @classmethod
    def load(cls, source: str | BinaryIO) -> 'AnestheticDistribution':
        from anesthetic import read_csv
        
        if isinstance(source, str):
            return AnestheticDistribution(read_csv(source))
        return cls.read(source)
    
    def param_names(self) -> list[str]:
        columns = self.nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]        
        return param_names
    
    def sample(self, key, sample_shape):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def icdf(self, u):
        raise NotImplementedError
