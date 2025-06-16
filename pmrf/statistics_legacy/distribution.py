import scipy.stats as stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from asteval import Interpreter

def string_to_scipy(str):
    if str == "" or str == None:
        return None
    try:
        name, args_str = str.split("(", 1)
        args_str = args_str.rstrip(")")
        aeval = Interpreter()
        args = aeval(f"({args_str})")
        rv = getattr(stats, name)(*map(float, args))
    except Exception as e:
        raise ValueError(f"Failed to create distribution instance for string {str}")
    
    return rv

def scipy_to_string(obj):
    if isinstance(obj, rv_continuous_frozen):
        name = obj.dist.name
        args = obj.args
        kwargs = obj.kwds
        # Combine args and sorted kwargs into one list for consistent formatting
        param_strs = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
        return f"{name}({', '.join(param_strs)})"
    else:
        return None  # or str(obj), or raise, depending on your use case
