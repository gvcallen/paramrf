from numpyro.distributions import Distribution, constraints

_BACKEND_REGISTRY = {}

try:
    from pmrf.distributions._backends.margarine import MargarineMAFAdapter
    _BACKEND_REGISTRY["margarine"] = MargarineMAFAdapter
except ImportError:
    pass

class MAFDistribution(Distribution):
    """
    A generic NumPyro-compatible distribution wrapper for MAF-based models.
    The actual implementation is delegated to a backend adapter.
    """
    support = constraints.real_vector
    has_rsample = False  # No reparameterization support

    def __init__(self, maf_or_path, backend="margarine", validate_args=None):
        if backend not in _BACKEND_REGISTRY:
            raise ValueError(f"Unsupported backend '{backend}'. Available: {list(_BACKEND_REGISTRY)}")

        adapter_cls = _BACKEND_REGISTRY[backend]
        if isinstance(maf_or_path, str):
            maf_model = adapter_cls.load(maf_or_path)
        else:
            maf_model = maf_or_path

        self.adapter = adapter_cls(maf_model)
        self.event_shape = (self.adapter.event_dim,)
        super().__init__(batch_shape=(), event_shape=self.event_shape, validate_args=validate_args)

    @classmethod
    def load(cls, path, backend="margarine", validate_args=None):
        """
        Load a MAFDistribution from a file using the specified backend.
        """
        adapter_cls = _BACKEND_REGISTRY[backend]
        maf_model = adapter_cls.load(path)
        return cls(maf_model, backend=backend, validate_args=validate_args)
    
    @classmethod
    def generate(cls, data, weights=None, backend="margarine", validate_args=None, **kwargs):
        adapter_cls = _BACKEND_REGISTRY[backend]
        maf_model = adapter_cls.generate(data, weights=weights, **kwargs)
        return cls(maf_model, backend=backend, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return self.adapter.sample(key, sample_shape)

    def log_prob(self, value):
        return self.adapter.log_prob(value)

    def icdf(self, u):
        return self.adapter.icdf(u)

    @property
    def min(self):
        return self.adapter.min

    @property
    def max(self):
        return self.adapter.max
