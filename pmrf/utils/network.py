import numpy as np
from scipy.interpolate import UnivariateSpline

def smooth_network(ntwk, smoothing=1e-4, k=3):
    """
    Smooth the S-parameters of a scikit-rf Network using spline smoothing.

    Parameters
    ----------
    ntwk : rf.Network
        Input network to smooth.
    smoothing : float
        Spline smoothing factor (larger -> smoother).
    k : int
        Spline order (default cubic).

    Returns
    -------
    rf.Network
        New network with smoothed S-parameters.
    """
    import skrf as rf
    
    freq = ntwk.frequency.f
    s = ntwk.s.copy()

    n_freq, n_ports, _ = s.shape
    s_smooth = np.zeros_like(s, dtype=complex)

    for i in range(n_ports):
        for j in range(n_ports):

            data = s[:, i, j]

            real_spline = UnivariateSpline(freq, np.real(data), s=smoothing, k=k)
            imag_spline = UnivariateSpline(freq, np.imag(data), s=smoothing, k=k)

            s_smooth[:, i, j] = real_spline(freq) + 1j * imag_spline(freq)

    new_ntwk = ntwk.copy()
    new_ntwk.s = s_smooth

    return new_ntwk