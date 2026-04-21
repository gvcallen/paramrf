from pmrf.optimize.minimize import ScipyMinimizer
import optimistix as optx

def is_optimizer(x):
    """
    Returns if a solver is suitable for frequentist optimization in :mod:`pmrf.optimize`.

    Returns `True` for :class:`pmrf.optimize.ScipyMinimizer` and :class:`optimistix.AbstractMinimiser`.
    """
    return isinstance(x, ScipyMinimizer | optx.AbstractMinimiser)