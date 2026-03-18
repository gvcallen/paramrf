"""
Test that we can build a simple RLC circuit and that its S-parameters match scikit-rf's.
"""
import numpy as np
from pmrf.core import Resistor, Inductor, ShuntCapacitor, SHORT
import pmrf as prf
from skrf.media import DefinedGammaZ0

def test_rlc():
    R = 100.0
    L = 20e-9
    C = 10e-10

    resistor = Resistor(R=R)
    inductor = Inductor(L=L)
    capacitor = ShuntCapacitor(C=C)

    freq = prf.Frequency(50, 200, 151, 'MHz')
    rlc = resistor ** inductor ** capacitor
    terminated_rlc_pmrf = rlc.terminated(SHORT)
    s_pmrf = terminated_rlc_pmrf.s(freq)

    freq = freq.to_skrf()
    media = DefinedGammaZ0(freq)
    terminated_rlc_skrf = media.resistor(R=R) ** media.inductor(L=L) ** media.shunt_capacitor(C=C) ** media.short()
    s_skrf = terminated_rlc_skrf.s

    assert np.allclose(s_pmrf, s_skrf)