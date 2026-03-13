import jax.numpy as jnp
from pmrf.constants import Number

def find_nearest_index(array: jnp.ndarray, value: Number) -> int:
    """
    Find the nearest index for a value in array.

    Parameters
    ----------
    array :  np.ndarray
        array we are searching for a value in
    value : element of the array
        value to search for

    Returns
    --------
    found_index : int
        the index at which the  numerically closest element to `value`
        was found at

    References
    ----------
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    """
    return (jnp.abs(array-value)).argmin()


def slice_domain(x: jnp.ndarray, domain: tuple):
    """
    Returns a slice object closest to the `domain` of `x`

    domain = x[slice_domain(x, (start, stop))]

    Parameters
    ----------
    vector : np.ndarray
        an array of values
    domain : tuple
        tuple of (start,stop) values defining the domain over
        which to slice

    Examples
    --------
    >>> x = linspace(0,10,101)
    >>> idx = slice_domain(x, (2,6))
    >>> x[idx]

    """
    start = find_nearest_index(x, domain[0])
    stop = find_nearest_index(x, domain[1])
    return slice(start, stop+1)