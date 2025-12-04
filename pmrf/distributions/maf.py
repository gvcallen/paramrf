from numpyro.distributions import Distribution, constraints

DEFAULT_BACKEND = 'flowjax'

def _get_backend_cls(backend_str: str):
    if backend_str == 'flowjax':
        from pmrf.distributions.flowjax import FlowJAXDistribution
    elif backend_str == 'margarine':
        from pmrf.distributions.margarine import MargarineMAFDistribution
    else:
        raise Exception(f'{backend_str} backend not supported by MAFDistribution')
    
def MAFDistribution(data, backend='flowjax') -> Distribution:
    if backend_str == 'flowjax':
        from pmrf.distributions.flowjax import FlowJAXDistribution
    elif backend_str == 'margarine':
        from pmrf.distributions.margarine import MargarineMAFDistribution
    else:
        raise Exception(f'{backend_str} backend not supported by MAFDistribution')    
        
class MAFDistribution(Distribution):
    """
    A generic NumPyro-compatible distribution wrapper for MAF-based models.
    The actual implementation is delegated to a backend adapter.
    """
    impl = None

    @classmethod
    def generate(cls, *args, backend=DEFAULT_BACKEND, **kwargs) -> Distribution:
        backend_cls = _get_backend_cls(backend)
        return backend_cls.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.backend, name)