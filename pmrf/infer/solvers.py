import inferix as infx

def is_inferer(x):
    """
    Returns if a solver is suitable for Bayesian inference in :mod:`pmrf.infer`.

    Returns `True` for :class:`infx.AbstractSampler`, :class:`infx.AbstractHostHypercubeNestedSampler`
    and :class:`infx.AbstractHostPhysicalNestedSampler`.
    """    
    return isinstance(x, infx.AbstractSampler | infx.AbstractHostHypercubeNestedSampler | infx.AbstractHostPhysicalNestedSampler)