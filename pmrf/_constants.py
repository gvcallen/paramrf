from typing import Callable, Literal

import pmrf.functions.math as mf

import pmrf.numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, PyTreeDef

ComponentFuncT = Literal["re", "im", "mag", "db", "db10", "rad", "deg", "arcl", "rad_unwrap", "deg_unwrap",
                         "arcl_unwrap", "vswr", "time", "time_db", "time_mag", "time_impulse", "time_step"]

PRIMARY_PROPERTIES = ('s', 'a')
FUNC_LOOKUP: dict[ComponentFuncT, tuple[str, Callable | None]] = {
    're': ('Real Part', np.real),
    'im': ('Imag Part', np.imag),
    'mag': ('Magnitude', np.abs),
    'db': ('Magnitude (dB)', mf.complex_2_db),
    'db10': ('Magnitude (dB)', mf.complex_2_db10),
    'rad': ('Phase (rad)', np.angle),
    'deg': ('Phase (deg)', lambda x: np.angle(x, deg=True)),
    'arcl': ('Arc Length',lambda x: np.angle(x) * np.abs(x)),
    'rad_unwrap': ('Phase (rad)', lambda x: mf.unwrap_rad(np.angle(x))),
    'deg_unwrap': ('Phase (deg)', lambda x: mf.radian_2_degree(mf.unwrap_rad(np.angle(x)))),
    'arcl_unwrap': ('Arc Length', lambda x: mf.unwrap_rad(np.angle(x)) * np.abs(x)),
    'vswr': ('VSWR', lambda x: (1 + abs(x)) / (1 - abs(x))),
    'time': ('Time (real)', mf.ifft),
    'time_db': ('Magnitude (dB)',  lambda x: mf.complex_2_db(mf.ifft(x))),
    'time_mag': ('Magnitude', lambda x: mf.complex_2_magnitude(mf.ifft(x))),
    'time_impulse': ('Magnitude', None),
    'time_step': ('Magnitude', None),
}