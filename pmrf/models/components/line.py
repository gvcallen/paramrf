"""
Transmission lines (RLGC, coaxial, microstrip)
"""
from abc import ABC, abstractmethod

import jax.numpy as jnp
from scipy.constants import c, mu_0, epsilon_0

from pmrf.math_functions import evaluate_bernstein_basis, evaluate_power_basis
from pmrf.rf_functions.conversions import renormalize_s
from pmrf.frequency import Frequency
from pmrf.parameters import Parameter
from pmrf.models.model import Model

class TransmissionLine(Model, ABC):
    """
    Abstract base class for all transmission line models.

    Provides the fundamental equations to construct S-parameters 
    based on frequency-dependent characteristic impedance ($Z_c$) 
    and total complex electrical length ($\gamma L$). Derived classes 
    must implement the `zc_gammaL` method.

    Attributes
    ----------
    floating : bool, default=False
        If True, modeled as a 4-port differential network (ports 0/2 and 1/3 
        form terminal pairs). If False, modeled as a 2-port single-ended network.
    """
    floating: bool = False 

    @abstractmethod
    def zc_gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates characteristic impedance ($Z_c$) and complex electrical length ($\gamma L$).

        Parameters
        ----------
        frequency : Frequency
            The frequency axis.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Array of characteristic impedance ($Z_c$) and complex electrical length ($\gamma L$).
        """
        raise NotImplementedError

    def s(self, frequency: Frequency) -> jnp.ndarray:
        zc, gL = self.zc_gammaL(frequency)
        
        if self.floating:
            denom = -1 + 9*jnp.exp(2*gL)
            s11 = (1 + 3*jnp.exp(2*gL)) / denom
            s12 = 4*jnp.exp(gL) / denom
            s13 = (-2 + 6*jnp.exp(2*gL)) / denom
            s14 = -s12

            s = jnp.array([
                [s11, s12, s13, s14],
                [s12, s11, s14, s13],
                [s13, s14, s11, s12],
                [s14, s13, s12, s11],
            ]).transpose(2, 0, 1)
        else:
            s11 = jnp.zeros(frequency.npoints, dtype=complex)
            s21 = jnp.exp(-1*gL)

            s = jnp.array([
                [s11, s21],
                [s21, s11],
            ]).transpose(2, 0, 1)

        # S-parameters are defined as traveling waves
        return renormalize_s(s, zc, self.z0, 'traveling', 'traveling')
    

class RLGCLine(TransmissionLine, ABC):
    """
    Abstract base class for a transmission line defined by its per-unit-length
    RLGC (Resistance, Inductance, Conductance, Capacitance) parameters.

    Derived classes must implement `rlgc` to define how these parameters
    behave over frequency.

    Attributes
    ----------
    length : Parameter, default=1.0
        Physical length of the line in meters.
    """
    length: Parameter = 1.0

    @abstractmethod
    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculates the frequency-dependent RLGC parameters.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.

        Returns
        -------
        tuple
            The R, L, G, and C parameter vectors.
        """
        raise NotImplementedError("'rlgc' must be implemented in the derived class")       

    def zc_gammaL(self, frequency: Frequency) -> jnp.ndarray:
        w = frequency.w
        R, L, G, C = self.rlgc(frequency)
        zc = jnp.sqrt((R + 1j*w*L) / (G + 1j*w*C))
        gamma = jnp.sqrt((R + 1j*w*L) * (G + 1j*w*C))
        gammaL = gamma*self.length
        
        return zc, gammaL  


class PhaseLine(TransmissionLine):
    """
    Ideal, lossless, and dispersionless transmission line defined by 
    electrical length at a reference frequency. Characteristic impedance 
    is real and constant; phase scales linearly.

    Example
    --------
    .. code-block:: python

        import pmrf as prf

        # Create an ideal 90-degree (quarter-wave) 50-ohm line at 1 GHz
        quarter_wave = prf.models.PhaseLine(
            zc=50.0,
            theta=90.0,
            f0=1e9
        )

        freq = prf.Frequency(start=0.5, stop=1.5, npoints=101, unit='ghz')
        s = quarter_wave.s(freq)

    Attributes
    ----------
    zc : Parameter, default=50.0
        Characteristic impedance in Ohms.
    theta : Parameter, default=90.0
        Electrical length (phase shift) in degrees at reference frequency `f0`.
    f0 : Parameter, default=1e9
        Reference frequency in Hz for `theta`.
    """
    zc: Parameter = 50.0    
    theta: Parameter = 90.0  
    f0: Parameter = 1e9

    def zc_gammaL(self, frequency: Frequency) -> jnp.ndarray:
        zc = self.zc * jnp.ones(frequency.npoints, dtype=complex)
        theta_rad = self.theta * jnp.pi / 180.0
        w0 = 2 * jnp.pi * self.f0
        beta_L = theta_rad * (frequency.w / w0)
        gammaL = 1j * beta_L
        
        return zc, gammaL


class ConstantRLGCLine(RLGCLine):
    """
    Transmission line with constant, frequency-independent RLGC parameters.

    Example
    --------
    .. code-block:: python

        import pmrf as prf

        lossless_line = prf.models.ConstantRLGCLine(
            L=368.8e-9,  # nH/m
            C=147.5e-12, # pF/m
            length=0.1   # 10 cm
        )

        freq = prf.Frequency(start=1, stop=5, npoints=101, unit='ghz')
        s = lossless_line.s(freq)

    Attributes
    ----------
    R : Parameter, default=0.0
        Resistance in Ohms/m.
    L : Parameter, default=280e-9
        Inductance in Henries/m.
    G : Parameter, default=0.0
        Conductance in Siemens/m.
    C : Parameter, default=90e-12
        Capacitance in Farads/m.
    """
    R: Parameter = 0.0
    L: Parameter = 280e-9
    G: Parameter = 0.0
    C: Parameter = 90e-12

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ones = jnp.ones(freq.npoints)
        return self.R * ones, self.L * ones, self.G * ones, self.C * ones
    

class PhysicalLine(RLGCLine):
    """
    Transmission line defined by nominal characteristic impedance, relative permittivity, 
    conductor attenuation, and dielectric loss tangent. 
    
    Equivalent to scikit-rf's `DefinedAEpTandZ0` wideband distortion model.

    Example
    --------
    .. code-block:: python

        import pmrf as prf

        line = prf.models.PhysicalLine(
            zn=50.0,
            length=1.0,
            epr=2.2,
            A=0.01,
            f_A=1.0,
            tand=0.001
        )

        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        s = line.s(freq)

    Attributes
    ----------
    zn : Parameter, default=50.0
        Nominal characteristic impedance defining the L/C ratio.
    epr : Parameter, default=1.0
        Relative permittivity.
    A : Parameter, default=0.0
        Conductor loss in dB/m/sqrt(Hz).
    f_A : Parameter, default=1.0
        Frequency scaling reference for attenuation in Hz.
    tand : Parameter, default=0.0
        Dielectric loss tangent.
    """
    zn: Parameter = 50.0
    epr: Parameter = 1.0
    A: Parameter = 0.0    
    f_A: Parameter = 1.0  
    tand: Parameter = 0.0 

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        w = freq.w
        f = w / (2 * jnp.pi)
        
        A_dB = self.A * jnp.sqrt(f / self.f_A)
        alpha_c = A_dB * (jnp.log(10) / 20.0)
        
        sqrt_epr = jnp.sqrt(self.epr)
        
        L_val = (self.zn * sqrt_epr) / c
        C_val = sqrt_epr / (self.zn * c)
        R_val = 2 * self.zn * alpha_c
        G_val = w * C_val * self.tand
        
        ones = jnp.ones(freq.npoints)
        R = R_val * ones
        L = L_val * ones
        G = G_val * ones
        C = C_val * ones
        
        return R, L, G, C    
    

class DatasheetLine(RLGCLine):
    """
    Transmission line defined by common datasheet parameters (nominal impedance, 
    dielectric constant, and loss factors). Includes skin effect (`k1`) and 
    dielectric loss (`k2`).

    Example
    --------
    .. code-block:: python

        import pmrf as prf

        cable = prf.models.DatasheetLine(
            zn=50.0,
            epr=2.1,
            k1=0.2,   # Skin effect loss factor
            k2=0.01,  # Dielectric loss factor
            length=1.0
        )

        freq = prf.Frequency(start=0.1, stop=10, npoints=201, unit='ghz')
        s = cable.s(freq)

    Attributes
    ----------
    zn : Parameter, default=50.0
        Nominal characteristic impedance.
    epr : Parameter, default=1.0
        Relative permittivity.
    k1 : Parameter, default=0.0
        Skin effect loss factor.
    k2 : Parameter, default=0.0
        Dielectric loss factor.
    epr_slope : Parameter | None, default=None
        Linear slope to apply to permittivity over the frequency bounds.
    loss_coeffs_normalized : bool, default=False
        If True, k1 and k2 are evaluated directly without normalizing to 100MHz references.
    freq_bounds : tuple | None, default=None
        Angular frequency limits (start, stop) used to scale `epr_slope`. Defaults to the analysis array bounds.
    """
    zn: Parameter = 50.0
    epr: Parameter = 1.0
    k1: Parameter = 0.0
    k2: Parameter = 0.0
    
    loss_coeffs_normalized: bool = False

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        w = freq.w
        zn, k1, k2 = self.zn, self.k1, self.k2

        epr = jnp.ones(w.shape[0]) * self.epr
        
        if not self.loss_coeffs_normalized:
            k1_norm = k1 * (1.0 / (100 * jnp.sqrt(2*jnp.pi * 10**6)))
            k2_norm = k2 * (1.0 / (100 * 2*jnp.pi * 10**6))
        else:
            k1_norm = k1
            k2_norm = k2

        sqrt_w = jnp.sqrt(w)
        dBtoNeper = jnp.log(10) / 20
        alpha_c = k1_norm * dBtoNeper * sqrt_w
        alpha_d = k2_norm * dBtoNeper * w
        sqrt_epr = jnp.sqrt(epr)
        
        R = 2*zn * alpha_c
        L = (zn * sqrt_epr) / c
        G = 2/zn * alpha_d
        C = (sqrt_epr) / (zn * c)
        
        return R, L, G, C
    

class CoaxialLine(RLGCLine):
    """
    Coaxial line defined directly by its physical geometry and material properties. 

    Example
    --------
    .. code-block:: python

        import pmrf as prf

        phys_cable = prf.models.CoaxialLine(
            din=0.9e-3,
            dout=2.95e-3,
            epr=1.5,
            tand=0.0004,
            rho=1.72e-8,
            length=0.5
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s_phys = phys_cable.s(freq)

    Attributes
    ----------
    din : Parameter, default=1.12e-3
        Inner conductor diameter in meters.
    dout : Parameter, default=3.2e-3
        Outer conductor inner diameter in meters.
    epr : Parameter, default=1.0
        Relative permittivity of the dielectric.
    mur : Parameter, default=1.0
        Relative permeability.
    tand : Parameter, default=0.0
        Loss tangent of the dielectric.
    rho : Parameter, default=1.68e-8
        Resistivity of the conductors in Ohm-meters.
    """
    din: Parameter = 1.12e-3
    dout: Parameter = 3.2e-3
    epr: Parameter = 1.0
    mur: Parameter = 1.0
    tand: Parameter = 0.0
    rho: Parameter = 1.68e-8
    
    @property
    def eps(self) -> jnp.ndarray:
        return epsilon_0 * self.epr * (1 - 1j * self.tand)
    
    @property
    def mu(self) -> jnp.ndarray:
        return mu_0 * self.mur
    
    def L_prime(self, freq: Frequency) -> jnp.ndarray:
        a, b = self.din / 2, self.dout / 2
        lnbOvera = jnp.log(b/a)
        return jnp.ones(freq.npoints) * self.mu / (2 * jnp.pi) * lnbOvera
    
    def C_prime(self, freq: Frequency) -> jnp.ndarray:
        a, b = self.din / 2, self.dout / 2
        lnbOvera = jnp.log(b/a)
        return jnp.ones(freq.npoints) * 2 * jnp.pi * jnp.real(self.eps) / lnbOvera
    
    def G_diel(self, freq: Frequency) -> jnp.ndarray:
        a, b = self.din / 2, self.dout / 2
        lnbOvera = jnp.log(b/a)
        return 2 * jnp.pi * freq.w * -jnp.imag(self.eps) / lnbOvera
        
    def R_skin(self, freq: Frequency) -> jnp.ndarray:
        return jnp.real(self.Z_skin(freq))
    
    def L_skin(self, freq: Frequency) -> jnp.ndarray:
        return jnp.imag(self.Z_skin(freq)) / freq.w
    
    def Z_skin(self, freq: Frequency):
        w, a, b, mu = freq.w, self.din / 2, self.dout / 2, self.mu
        sigma_a, sigma_b = 1 / self.rho, 1 / self.rho
        
        L_skin_a = (1 / (2 * jnp.pi * a)) * jnp.sqrt(mu / (2 * w * sigma_a))
        L_skin_b = (1 / (2 * jnp.pi * b)) * jnp.sqrt(mu / (2 * w * sigma_b))
        L_skin = L_skin_a + L_skin_b
                
        R_skin_a = (1 / (2 * jnp.pi * a)) * jnp.sqrt(w * mu / (2 * sigma_a))
        R_skin_b = (1 / (2 * jnp.pi * b)) * jnp.sqrt(w * mu / (2 * sigma_b))
        R_skin = R_skin_a + R_skin_b            
        
        return R_skin + 1j * w * L_skin

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:        
        L = self.L_prime(freq) + self.L_skin(freq)
        C = self.C_prime(freq)
        G = self.G_diel(freq)
        R = self.R_skin(freq)
        
        return R, L, G, C
    
    
class MicrostripLine(RLGCLine):
    """
    Microstrip line defined by standard geometric and material properties.
    
    Relies on standard Wheeler approximations. Note that configurations where 
    height > width (h > w) are not yet supported.

    Attributes
    ----------
    w : Parameter, default=3e-3
        Width of the microstrip trace in meters.
    h : Parameter, default=1.6e-3
        Height of the dielectric substrate in meters.
    epr : Parameter, default=4.3
        Relative permittivity of the dielectric substrate.
    tand : Parameter, default=0.0
        Dielectric loss tangent.
    rho : Parameter, default=0.0
        Resistivity of the conductor trace and ground plane in Ohm-meters.
    """
    w: Parameter = 3e-3
    h: Parameter = 1.6e-3
    epr: Parameter = 4.3
    tand: Parameter = 0.0
    rho: Parameter = 0.0

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        W, H = self.w, self.h
        epr, tand, rho = self.epr, self.tand, self.rho
        
        u = W / H

        t1 = ((epr + 1) / 2)
        t2 = ((epr - 1) / 2)
        t3 = 1 / jnp.sqrt(1 + 12 / u)
        epe = (t1 + t2*t3) * jnp.ones(freq.npoints)
        
        Za = (120 * jnp.pi) / (u + 1.393 + 0.667 * jnp.log(u + 1.444))
        Ze = Za / jnp.sqrt(epe)

        L = (Ze * jnp.sqrt(epe)) / c
        C = (jnp.sqrt(epe)) / (Ze * c)
        R = (1 / W) * jnp.sqrt(2 * mu_0 * rho) * jnp.sqrt(freq.w)
        G = (1 / (Za * c)) * (epr * (epe - 1) / (epr - 1)) * tand * freq.w
        
        return R, L, G, C
    

# Not currently exported but kept for reference
class _DispersiveCoaxialLine(RLGCLine):
    """
    Coaxial line defined directly by its physical geometry and material properties,


    Material properties can be modeled as frequency-dependent polynomials:
    - **'constant'**: (Default) Scalar value.
    - **'ppoly'**: Power basis polynomial (value is list of coefficients).
    - **'bpoly'**: Bernstein basis polynomial (value is list of coefficients).

    Example
    --------
    .. code-block:: python

        import pmrf as prf

        phys_cable = prf.models.CoaxialLine(
            din=0.9e-3,
            dout=2.95e-3,
            epr=[2.1, 2.05], # Linear taper using Bernstein
            epr_model='bpoly',
            tand=0.0004,
            rho=1.72e-8,
            length=0.5
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s_phys = phys_cable.s(freq)

    Attributes
    ----------
    din : Parameter, default=1.12e-3
        Inner conductor diameter in meters.
    dout : Parameter, default=3.2e-3
        Outer conductor inner diameter in meters.
    epr : Parameter, default=1.0
        Relative permittivity of the dielectric.
    mur : Parameter, default=1.0
        Relative permeability.
    tand : Parameter, default=0.0
        Loss tangent of the dielectric.
    rho : Parameter, default=1.68e-8
        Resistivity of the conductors in Ohm-meters.
    epr_model : str, default='constant'
        Polynomial model type for `epr`.
    mur_model : str, default='constant'
        Polynomial model type for `mur`.
    tand_model : str, default='constant'
        Polynomial model type for `tand`.
    rho_model : str, default='constant'
        Polynomial model type for `rho`.
    separate_rho : bool, default=False
        If True, queries distinct `rhoin` and `rhoout` properties for the conductors.
    neglect_skin_inductance : bool, default=False
        If True, excludes internal skin effect inductance from total L.
    """
    din: Parameter = 1.12e-3
    dout: Parameter = 3.2e-3
    epr: Parameter = 1.0
    mur: Parameter = 1.0
    tand: Parameter = 0.0
    rho: Parameter = 1.68e-8
    
    epr_model: str = 'constant'
    mur_model: str = 'constant'
    tand_model: str = 'constant'
    rho_model: str = 'constant'
    separate_rho: bool = False
    neglect_skin_inductance: bool = False

    def interpolated(self, param: str, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        if param.startswith('rho'):
            model = str(getattr(self, 'rho_model'))
        else:
            model = str(getattr(self, f'{param}_model'))
        
        if model == 'constant':
            value = getattr(self, param) * jnp.ones(w.shape[0])
        else:
            coeffs = getattr(self, param)
            if model.startswith('ppoly'):
                value = evaluate_power_basis(w, coeffs, w[0], w[-1])
            else:
                value = evaluate_bernstein_basis(w, coeffs, w[0], w[-1])
                
        return value
            
    def epr_f(self, freq: Frequency) -> jnp.ndarray:
        return self.interpolated('epr', freq)
    
    def tand_f(self, freq: Frequency) -> jnp.ndarray:
        return self.interpolated('tand', freq)
    
    def mur_f(self, freq: Frequency) -> jnp.ndarray:
        return self.interpolated('mur', freq)
    
    def rho_f(self, freq: Frequency) -> jnp.ndarray:
        return self.interpolated('rho', freq)
    
    def rhoin_f(self, freq: Frequency) -> jnp.ndarray:
        return self.interpolated('rhoin', freq) if self.separate_rho else self.rho_f(freq)
    
    def rhoout_f(self, freq: Frequency) -> jnp.ndarray:
        return self.interpolated('rhoout', freq) if self.separate_rho else self.rho_f(freq)
    
    def eps_f(self, freq: Frequency) -> jnp.ndarray:
        return epsilon_0 * self.epr_f(freq) * (1 - 1j * self.tand_f(freq))
    
    def mu_f(self, freq: Frequency) -> jnp.ndarray:
        return mu_0 * self.mur_f(freq)
    
    def L_prime(self, freq: Frequency) -> jnp.ndarray:
        a, b = self.din / 2, self.dout / 2
        lnbOvera = jnp.log(b/a)
        return self.mu_f(freq) / (2 * jnp.pi) * lnbOvera
    
    def C_prime(self, freq: Frequency) -> jnp.ndarray:
        a, b = self.din / 2, self.dout / 2
        lnbOvera = jnp.log(b/a)
        return 2 * jnp.pi * jnp.real(self.eps_f(freq)) / lnbOvera
    
    def G_diel(self, freq: Frequency) -> jnp.ndarray:
        a, b = self.din / 2, self.dout / 2
        lnbOvera = jnp.log(b/a)
        return 2 * jnp.pi * freq.w * -jnp.imag(self.eps_f(freq)) / lnbOvera
        
    def R_skin(self, freq: Frequency) -> jnp.ndarray:
        return jnp.real(self.Z_skin(freq))
    
    def L_skin(self, freq: Frequency) -> jnp.ndarray:
        return jnp.imag(self.Z_skin(freq)) / freq.w
    
    def Z_skin(self, freq: Frequency):
        w, a, b, mu = freq.w, self.din / 2, self.dout / 2, self.mu_f(freq)
        sigma_a, sigma_b = 1 / self.rhoin_f(freq), 1 / self.rhoout_f(freq)
        
        L_skin_a = (1 / (2 * jnp.pi * a)) * jnp.sqrt(mu / (2 * w * sigma_a))
        L_skin_b = (1 / (2 * jnp.pi * b)) * jnp.sqrt(mu / (2 * w * sigma_b))
        L_skin = L_skin_a + L_skin_b
                
        R_skin_a = (1 / (2 * jnp.pi * a)) * jnp.sqrt(w * mu / (2 * sigma_a))
        R_skin_b = (1 / (2 * jnp.pi * b)) * jnp.sqrt(w * mu / (2 * sigma_b))
        R_skin = R_skin_a + R_skin_b            
        
        return R_skin + 1j * w * L_skin

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:        
        if not self.neglect_skin_inductance:
            L = self.L_prime(freq) + self.L_skin(freq)
        else:
            L = self.L_prime(freq)
        
        C = self.C_prime(freq)
        G = self.G_diel(freq)
        R = self.R_skin(freq)
        
        return R, L, G, C