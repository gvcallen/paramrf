"""
Non-ideal models (e.g. resistors with parasitics)
"""
import jax.numpy as jnp

from pmrf.core import Model, Frequency
from pmrf.models.components.lumped import Resistor
from pmrf.models.components.topological import PiCLC


class CLCResistor(Model):
    """
    A model for a non-ideal resistor with parasitic capacitance and inductance.

    This model represents a physical resistor as a parasitic Pi-network 
    (Capacitor-Inductor-Capacitor) cascaded with an ideal resistive element. 
    This topology is common for modeling SMD resistors at high frequencies.

    Attributes
    ----------
    res : Resistor
        The ideal resistor model.
    clc : PiCLC
        The parasitic Pi-network model (C-L-C).

    Examples
    --------
    .. code-block:: python
    
        import pmrf as prf

        # Create a model for a 100-ohm resistor with parasitics
        non_ideal_r = prf.models.CLCResistor(
            res=prf.models.Resistor(R=100),
            clc=prf.models.PiCLC(C1=0.05e-12, L=0.1e-9, C2=0.05e-12)
        )

        # You can access the ideal and parasitic parts
        print(f"Ideal Resistance: {non_ideal_r.ideal.R.value} Ohms")

        # Calculate the S-parameters of the complete non-ideal model
        freq = prf.Frequency(start=0.1, stop=20, npoints=201, unit='ghz')
        s = non_ideal_r.s(freq)

        print(f"S11 at 10 GHz: {s[freq.center_idx, 0, 0]:.2f}")
    """
    res: Resistor = Resistor()
    clc: PiCLC = PiCLC()

    @property
    def ideal(self) -> Model:
        """Model: The ideal resistor component."""
        return self.res
    
    @property
    def parasitics(self) -> Model:
        """Model: The parasitic C-L-C network."""
        return self.clc
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        cascaded = self.clc ** self.res
        return cascaded.a(freq)