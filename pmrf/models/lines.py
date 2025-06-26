from abc import abstractmethod

import pmrf.numpy as np
from scipy.constants import c, mu_0, epsilon_0
from dataclasses import fields

from pmrf.functions.math import evaluate_bernstein_basis, evaluate_power_basis
from pmrf._frequency import Frequency
from pmrf.parameters import Parameter
from pmrf._model import Model
from pmrf._misc import field

class RLGCLine(Model):
    """
    **Overview**

    An abstract base class for a transmission line defined by its per-unit-length
    RLGC (Resistance, Inductance, G, Conductance) parameters.

    This class provides the fundamental equations to calculate the ABCD-matrix
    of a transmission line given its propagation constant ($\gamma$) and
    characteristic impedance ($Z_c$), which are derived from the RLGC values.

    Derived classes must implement the `rlgc` method, which defines how the
    four distributed parameters (R, L, G, C) behave as a function of frequency.
    """
    length: Parameter = 1.0

    @abstractmethod
    def rlgc(self, freq: Frequency) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates the frequency-dependent RLGC parameters of the line.

        This method must be implemented by any derived concrete class.

        Args:
            freq (Frequency): The frequency axis at which to evaluate the parameters.

        Returns:
            tuple: A tuple containing the R, L, G, and C parameter vectors, in that order.
        """
        raise NotImplementedError("'rlgc' must be implemented in the derived class")

    def a(self, freq: Frequency) -> np.ndarray:
        """Calculates the ABCD-matrix from the line's RLGC parameters.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant ABCD-matrix.
        """
        w = freq.w
        R, L, G, C = self.rlgc(freq)
        gamma = np.sqrt((R + 1j*w*L) * (G + 1j*w*C))
        Zc = np.sqrt((R + 1j*w*L) / (G + 1j*w*C))

        gL = gamma*self.length
        
        a = np.array([
            [np.cosh(gL), Zc * np.sinh(gL)],
            [1 / Zc * np.sinh(gL), np.cosh(gL)]
        ]).transpose(2, 0, 1)

        return a
    
class ConstantRLGCLine(RLGCLine):
    """
    **Overview**

    An RLGC line with constant, frequency-independent per-unit-length parameters.

    This is the simplest transmission line model, where R, L, G, and C
    are single scalar values.

    **Example**

    ```python
    import pmrf as prf

    # Create a lossless 50-ohm line model
    # L and C are chosen to produce Z0=50 and epr=2.2
    lossless_line = prf.models.ConstantRLGCLine(
        L=368.8e-9, # nH/m
        C=147.5e-12, # pF/m
        length=0.1 # 10 cm
    )

    # Calculate its S-parameters over a frequency range
    freq = prf.Frequency(start=1, stop=5, npoints=101, unit='ghz')
    s = lossless_line.s(freq)

    print(f"S21 magnitude at center frequency: {abs(s[freq.center_idx, 1, 0]):.3f}")
    ```
    """
    R: Parameter = 0.0
    L: Parameter = 280e-9
    G: Parameter = 0.0
    C: Parameter = 90e-12

    def rlgc(self, _: Frequency) -> tuple[Parameter, Parameter, Parameter, Parameter]:
        """Returns the constant RLGC parameters.

        Args:
            _ (Frequency): The frequency axis (ignored).

        Returns:
            tuple: The constant R, L, G, and C parameters.
        """
        return self.R, self.L, self.G, self.C
    
class DatasheetCoaxial(RLGCLine):
    """
    **Overview**

    A coaxial line defined by parameters typically found on a datasheet.

    This model provides a convenient way to define a coaxial cable from its
    nominal impedance, dielectric constant, and loss factors, rather than
    from the fundamental RLGC values directly. It includes terms for both
    skin effect loss (`k1`) and dielectric loss (`k2`).

    **Example**

    ```python
    import pmrf as prf
    import numpy as np

    # Define a 1m coaxial cable from typical datasheet values
    cable = prf.models.DatasheetCoaxial(
        zn=50.0,
        epr=2.1,
        k1=0.2, # Skin effect loss factor
        k2=0.01, # Dielectric loss factor
        length=1.0
    )

    freq = prf.Frequency(start=0.1, stop=10, npoints=201, unit='ghz')
    s = cable.s(freq)

    # Plot the insertion loss
    # import matplotlib.pyplot as plt
    # plt.plot(freq.f_scaled, 20*np.log10(abs(s[:,1,0])))
    # plt.xlabel(f"Frequency ({freq.unit})")
    # plt.ylabel("Insertion Loss (dB)")
    # plt.grid(True)
    # plt.show()
    ```
    """
    zn: Parameter = 50.0
    epr: Parameter = 1.0
    epr_slope: Parameter | None = None
    k1: Parameter = 0.0
    k2: Parameter = 0.0

    loss_coeffs_normalized: bool = field(default=False, static=True)
    freq_bounds: tuple | None = field(default=None, static=True)

    def rlgc(self, freq: Frequency) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates RLGC parameters from datasheet values.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            tuple: The calculated R, L, G, and C parameter vectors.
        """
        w = freq.w
        zn, k1, k2 = self.zn, self.k1, self.k2

        if self.freq_bounds is not None:
            w_start, w_stop = self.freq_bounds
        else:
            w_start, w_stop = w[0], w[-1]

        wn = (w - w_start) / (w_stop - w_start)
        if self.epr_slope is None:
            epr = np.ones(w.shape[0]) * self.epr
        else:
            epr = np.ones(w.shape[0]) * self.epr + self.epr_slope * wn
        
        if not self.loss_coeffs_normalized:
            k1_norm = k1 * (1.0 / (100 * np.sqrt(2*np.pi * 10**6)))
            k2_norm = k2 * (1.0 / (100 * 2*np.pi * 10**6))
        else:
            k1_norm = k1
            k2_norm = k2

        sqrt_w = np.sqrt(w)
        dBtoNeper = np.log(10) / 20
        alpha_c = k1_norm * dBtoNeper * sqrt_w
        alpha_d = k2_norm * dBtoNeper * w
        sqrt_epr = np.sqrt(epr)
        
        R = 2*zn * alpha_c
        L = (zn * sqrt_epr) / c
        G = 2/zn * alpha_d
        C = (sqrt_epr) / (zn * c)
        return R, L, G, C
        
class PhysicalCoaxial(RLGCLine):
    """
    **Overview**

    A coaxial line defined directly by its physical and material properties.

    This model calculates the RLGC parameters from the geometry (inner/outer
    diameters) and material properties (permittivity, permeability, loss tangent,
    resistivity). It provides a high-fidelity model that accounts for the skin
    effect in the conductors and frequency-dependent material properties.

    Several material properties can be defined as frequency-dependent polynomials
    using the `_model` arguments (e.g., `epr_model`). The available models are:
    - **'constant'**: (Default) The parameter is a single scalar value.
    - **'ppoly'**: The parameter is a polynomial in the power basis. The `Parameter`
      value should be a list of coefficients.
    - **'bpoly'**: The parameter is a polynomial in the Bernstein basis. The `Parameter`
      value should be a list of coefficients.

    **Example**

    ```python
    import pmrf as prf

    # A coaxial cable where the dielectric constant (epr) varies with frequency
    # We model epr as a 1st-order Bernstein polynomial (a line)
    # The coefficients are the start and end values of the line.
    phys_cable = prf.models.PhysicalCoaxial(
        din=0.9e-3,
        dout=2.95e-3,
        epr=[2.1, 2.05], # epr goes from 2.1 to 2.05 over the freq range
        epr_model='bpoly',
        tand=0.0004,
        rho=1.72e-8, # Copper resistivity
        length=0.5
    )

    freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
    s_phys = phys_cable.s(freq)
    ```
    """
    din: Parameter = 1.12e-3
    dout: Parameter = 3.2e-3
    epr: Parameter = 1.0
    mur: Parameter = 1.0
    tand: Parameter = 0.0
    rho: Parameter = 1.68e-8
    
    epr_model: str = field(default='constant', static=True)
    mur_model: str = field(default='constant', static=True)
    tand_model: str = field(default='constant', static=True)
    rho_model: str = field(default='constant', static=True)
    separate_rho: bool = field(default=False, static=True)
    neglect_skin_inductance: bool = field(default=False, static=True)

    def __init__(self, **kwargs):
        # This custom initializer allows setting polynomial orders via kwargs
        # like `epr_order=1`, and correctly initializes coefficient arrays if needed.
        init_fields = {f.name for f in fields(self)}
        
        # Separate kwargs for fields vs other options
        field_kwargs = {k: v for k, v in kwargs.items() if k in init_fields}
        order_kwargs = {k: v for k, v in kwargs.items() if k not in init_fields}

        super().__init__(**field_kwargs)
        
        poly_params = ['epr', 'mur', 'tand', 'rho']
        for param in poly_params:
            model = getattr(self, f'{param}_model')
            if model != 'constant':
                current = np.array(getattr(self, param))
                order = order_kwargs.get(f'{param}_order', 1)
                
                if np.isscalar(current):
                    if model == 'bpoly':
                        coeffs = [current] * (order+1)
                    elif model == 'ppoly':
                        coeffs = [current] + [0.0]*order
                    else:
                        raise Exception(f"Unknown frequency model for parameter {param}")
                    # Use object.__setattr__ to modify the frozen dataclass field
                    object.__setattr__(self, param, coeffs)

    def interpolated(self, param: str, freq: Frequency) -> np.ndarray:
        """Evaluates a potentially frequency-dependent parameter.

        Args:
            param (str): The name of the parameter to evaluate (e.g., 'epr').
            freq (Frequency): The frequency axis for the evaluation.

        Returns:
            np.ndarray: The parameter's value across the frequency axis.
        """
        w = freq.w
        if param.startswith('rho'):
            model = str(getattr(self, 'rho_model'))
        else:
            model = str(getattr(self, f'{param}_model'))
        
        if model == 'constant':
            value = getattr(self, param) * np.ones(w.shape[0])
        else:
            coeffs = getattr(self, param)
            if model.startswith('ppoly'):
                value = evaluate_power_basis(w, coeffs, w[0], w[-1])
            else:
                value = evaluate_bernstein_basis(w, coeffs, w[0], w[-1])
                
        return value
            
    def epr_f(self, freq: Frequency) -> np.ndarray:
        """The relative permittivity ($\epsilon_r$) as a function of frequency."""
        return self.interpolated('epr', freq)
    
    def tand_f(self, freq: Frequency) -> np.ndarray:
        """The loss tangent (tan$\delta$) as a function of frequency."""
        return self.interpolated('tand', freq)
    
    def mur_f(self, freq: Frequency) -> np.ndarray:
        """The relative permeability ($\mu_r$) as a function of frequency."""
        return self.interpolated('mur', freq)
    
    def rho_f(self, freq: Frequency) -> np.ndarray:
        """The conductor resistivity ($\rho$) as a function of frequency."""
        return self.interpolated('rho', freq)
    
    def rhoin_f(self, freq: Frequency) -> np.ndarray:
        """The inner conductor resistivity as a function of frequency."""
        return self.interpolated('rhoin', freq) if self.separate_rho else self.rho_f(freq)
    
    def rhoout_f(self, freq: Frequency) -> np.ndarray:
        """The outer conductor resistivity as a function of frequency."""
        return self.interpolated('rhoout', freq) if self.separate_rho else self.rho_f(freq)
    
    def eps_f(self, freq: Frequency) -> np.ndarray:
        """The complex permittivity ($\epsilon$) as a function of frequency."""
        return epsilon_0 * self.epr_f(freq) * (1 - 1j * self.tand_f(freq))
    
    def mu_f(self, freq: Frequency) -> np.ndarray:
        """The complex permeability ($\mu$) as a function of frequency."""
        return mu_0 * self.mur_f(freq)
    
    def L_prime(self, freq: Frequency) -> np.ndarray:
        """The per-unit-length external inductance."""
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return self.mu_f(freq) / (2 * np.pi) * lnbOvera
    
    def C_prime(self, freq: Frequency) -> np.ndarray:
        """The per-unit-length capacitance."""
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return 2 * np.pi * np.real(self.eps_f(freq)) / lnbOvera
    
    def G_diel(self, freq: Frequency) -> np.ndarray:
        """The per-unit-length dielectric conductance."""
        a, b = self.din / 2, self.dout / 2
        lnbOvera = np.log(b/a)
        return 2 * np.pi * freq.w * -np.imag(self.eps_f(freq)) / lnbOvera
        
    def R_skin(self, freq: Frequency) -> np.ndarray:
        """The per-unit-length resistance due to skin effect."""
        return np.real(self.Z_skin(freq))
    
    def L_skin(self, freq: Frequency) -> np.ndarray:
        """The per-unit-length internal inductance due to skin effect."""
        return np.imag(self.Z_skin(freq)) / freq.w
    
    def Z_skin(self, freq: Frequency) -> np.ndarray:
        """The per-unit-length internal impedance due to skin effect."""
        w = freq.w
        a = self.din / 2
        b = self.dout / 2
        mu = self.mu_f(freq)
        sigma_a = 1 / self.rhoin_f(freq)
        sigma_b = 1 / self.rhoout_f(freq)
        
        # Using the simplified formula for Z_surface
        Z_s_a = (1 + 1j) / (sigma_a * (np.sqrt(2 / (w * mu * sigma_a))))
        Z_s_b = (1 + 1j) / (sigma_b * (np.sqrt(2 / (w * mu * sigma_b))))

        return (1 / (2*np.pi)) * ((Z_s_a / a) + (Z_s_b / b))

    def rlgc(self, freq: Frequency) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates RLGC parameters from physical and material properties.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            tuple: The calculated R, L, G, and C parameter vectors.
        """
        if not self.neglect_skin_inductance:
            L = self.L_prime(freq) + self.L_skin(freq)
        else:
            L = self.L_prime(freq)
        
        C = self.C_prime(freq)
        G = self.G_diel(freq)
        R = self.R_skin(freq)
        
        return R, L, G, C