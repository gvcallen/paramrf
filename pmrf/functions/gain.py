import pmrf.numpy as np
from pmrf._model import Model
from pmrf._frequency import Frequency

# The available gain of a two-port network relative to the available power of a source, where port 1 of the two-port is connected to the source
def available_gain(source: Model, passive: Model, freq: Frequency, source_port=0):
    if not source_port == 0:
        passive = passive.flipped()
    
    # Note: Equation 10.25 on page 507 of Pozar, using the references variable names from Pozar
    ntwk_back = passive.flipped() ** source
    
    gamma_S = source.s(freq)[:, 0, 0]

    gamma_out = ntwk_back.s(freq)[:, 0, 0]
    s21, s11 = passive.s(freq)[:, 1, 0], passive.s(freq)[:, 0, 0]
    
    num = np.abs(s21)**2 * (1 - np.abs(gamma_S)**2)
    den = np.abs(1 - s11*gamma_S)**2 * (1 - np.abs(gamma_out)**2)
    Gav = num / den

    return Gav