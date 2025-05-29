import numpy as np

def dB20(values):
    return 20 * np.log10(np.abs(values))

def dB20inv(values):
    return 10 ** (values / 20)

def p2r(radii, angles, deg=False):
    if deg:
        angles = np.deg2rad(angles)
    return radii * np.exp(1j*angles)

def r2p(x, deg=False):
    return abs(x), np.angle(x, deg=deg)

def rms(values):
    return np.sqrt(np.mean(values**2))

def norm(y: np.ndarray, mode='L2', axis=None):
    if mode == 'Linf':
        y = np.linalg.norm(y, ord=np.inf, axis=axis)        # max(abs(y))
    elif mode == 'L1':
        y = np.linalg.norm(y, ord=1, axis=axis)             # sum(abs(y))
    elif mode == 'L2':
        y = np.linalg.norm(y, ord=2, axis=axis)             # sqrt(sum(abs(y)**2))
    elif mode == 'L2sqr':
        y = np.linalg.norm(y, ord=2, axis=axis)**2          # sum(abs(y)**2)
    else:
        raise Exception('Unknown norm type')
    
    return y

def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)