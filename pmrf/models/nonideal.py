from abc import abstractmethod

import jax.numpy as jnp

from pmrf.models.lumped import Resistor
from pmrf.models.topological import PiCLC
from pmrf._model import Model
from pmrf._frequency import Frequency

class NonIdealResistor(Model):
    """
    **Overview**

    An abstract base class for creating realistic resistor models that include
    parasitic effects.

    This class provides a framework for representing a physical resistor as a
    combination of an ideal resistive element and a network of parasitic
    components (like series inductance and parallel capacitance).

    Subclasses are required to implement the `ideal` and `parasitics`
    properties to define the specific topology of the non-ideal model.
    """
    @property
    @abstractmethod
    def ideal(self) -> Model:
        """
        An abstract property representing the ideal part of the component.

        Returns:
            Model: The model for the ideal resistor.
        """
        raise Exception("Subclasses must implement the 'ideal' property.")

    @property
    @abstractmethod
    def parasitics(self) -> Model:
        """
        An abstract property representing the parasitic network of the component.

        Returns:
            Model: The model for the parasitic network.
        """
        raise Exception("Subclasses must implement the 'parasitics' property.")

class CLCResistor(NonIdealResistor):
    """
    **Overview**

    A model for a non-ideal resistor with parasitic capacitance and inductance.

    This model represents a physical resistor as an ideal resistive element
    cascaded with a Pi-network (Capacitor-Inductor-Capacitor). This topology
    is common for modeling SMD resistors at high frequencies.

    **Example**

    ```python
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
    ```
    """
    res: Resistor = Resistor()
    clc: PiCLC = PiCLC()

    @property
    def ideal(self) -> Model:
        """The ideal `Resistor` part of the model.

        Returns:
            Model: The ideal resistor component.
        """
        return self.res
    
    @property
    def parasitics(self) -> Model:
        """The parasitic `PiCLC` network part of the model.

        Returns:
            Model: The PiCLC parasitic network.
        """
        return self.clc
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the ABCD-matrix of the complete non-ideal resistor model.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant ABCD-matrix.
        """
        cascaded = self.clc ** self.res
        return cascaded.a(freq)