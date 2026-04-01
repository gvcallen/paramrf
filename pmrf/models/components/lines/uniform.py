"""
Uniform transmission lines (RLGC, coaxial, microstrip)
"""
from abc import ABC, abstractmethod

from scipy.constants import c, mu_0, epsilon_0
import jax.numpy as jnp
import parax as prx
from parax import Parameter

from pmrf.core import Frequency, Model
from pmrf.rf import renormalize_s

class TransmissionLine(Model, ABC):
    r"""
    Abstract base class for all uniform transmission line models.

    Provides the fundamental equations to construct S-parameters 
    based on frequency-dependent characteristic impedance ($Z_c$) 
    and total complex electrical length ($\gamma L$). Derived classes 
    must implement the `zc_gammaL` method.

    **Mathematical Formulation**

    For a single-ended 2-port transmission line, the traveling wave S-parameters with respect to $Z_c$ are:
    $$S_{11} = S_{22} = 0$$
    $$S_{21} = S_{12} = e^{-\gamma L}$$

    This model computes these S-parameters and then re-normalized them into $Z_0$ and the power-wave definition
    using :meth:`pmrf.rf.renormalize_s`.

    Attributes
    ----------
    floating : bool, default=False
        If True, modeled as a 4-port differential network (ports 0/1 and 2/3 
        form terminal pairs). If False, modeled as a 2-port single-ended network.
    """
    floating: bool = prx.field(default=False, static=True)

    @abstractmethod
    def zc_and_gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""
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
        zc, gL = self.zc_and_gammaL(frequency)
        
        if self.floating:
            denom = -1 + 9*jnp.exp(2*gL)
            a = (1 + 3*jnp.exp(2*gL)) / denom
            b = 4*jnp.exp(gL) / denom
            c = (-2 + 6*jnp.exp(2*gL)) / denom
            d = -b

            s = jnp.array([
                [a, c, b, d],
                [c, a, d, b],
                [b, d, a, c],
                [d, b, c, a],
            ]).transpose(2, 0, 1)
        else:
            a = jnp.zeros(frequency.npoints, dtype=complex)
            s21 = jnp.exp(-1*gL)

            s = jnp.array([
                [a, s21],
                [s21, a],
            ]).transpose(2, 0, 1)

        # Renormalize into the model's characteristic impedance and power waves
        # (the above formulation is in terms of traveling waves).
        return renormalize_s(s, zc, self.z0, 'traveling', 'power')
    

class RLGCLine(TransmissionLine, ABC):
    r"""
    Abstract base class for a transmission line defined by its per-unit-length
    RLGC (Resistance, Inductance, Conductance, Capacitance) parameters.

    Derived classes must implement `rlgc` to define how these parameters
    behave over frequency.

    **Mathematical Formulation**

    The characteristic impedance ($Z_c$) and complex propagation constant ($\gamma$) are derived as:
    $$Z_c = \sqrt{\frac{R + j\omega L}{G + j\omega C}}$$
    $$\gamma = \sqrt{(R + j\omega L)(G + j\omega C)}$$

    The total complex electrical length is $\gamma L$.

    Attributes
    ----------
    length : Parameter, default=1.0
        Physical length of the line in meters.
    """
    length: Parameter = 1.0

    @abstractmethod
    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""
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

    def zc_and_gammaL(self, frequency: Frequency) -> jnp.ndarray:
        w = frequency.w
        R, L, G, C = self.rlgc(frequency)
        zc = jnp.sqrt((R + 1j*w*L) / (G + 1j*w*C))
        gamma = jnp.sqrt((R + 1j*w*L) * (G + 1j*w*C))
        gammaL = gamma*self.length
        
        return zc, gammaL  


class PhaseLine(TransmissionLine):
    r"""
    Ideal, lossless, and dispersionless transmission line defined by 
    electrical length at a reference frequency. Characteristic impedance 
    is real and constant; phase scales linearly.

    **Mathematical Formulation**

    $$Z_c(\omega) = z_c$$
    $$\gamma L(\omega) = j \cdot \left(\theta \cdot \frac{\pi}{180}\right) \cdot \frac{\omega}{\omega_0}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.core import PhaseLine

        # Create an ideal 90-degree (quarter-wave) 50-ohm line at 1 GHz
        quarter_wave = PhaseLine(
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
    theta: Parameter = 90.0
    zc: Parameter = 50.0    
    
    f0: float = 1e9

    def zc_and_gammaL(self, frequency: Frequency) -> jnp.ndarray:
        zc = self.zc * jnp.ones(frequency.npoints, dtype=complex)
        theta_rad = self.theta * jnp.pi / 180.0
        w0 = 2 * jnp.pi * self.f0
        beta_L = theta_rad * (frequency.w / w0)
        gammaL = 1j * beta_L
        
        return zc, gammaL


class ConstantRLGCLine(RLGCLine):
    r"""
    Transmission line with constant, frequency-independent RLGC parameters.

    **Mathematical Formulation**

    $$R(\omega) = R$$
    $$L(\omega) = L$$
    $$G(\omega) = G$$
    $$C(\omega) = C$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.core import ConstantRLGCLine

        lossless_line = ConstantRLGCLine(
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
    r"""
    Transmission line defined by nominal characteristic impedance, relative permittivity, 
    conductor attenuation, and dielectric loss tangent.
    
    **Mathematical Formulation**

    The frequency-dependent attenuation components are computed as:
    $$\alpha_c = A \cdot \sqrt{\frac{f}{fA}} \cdot \frac{\ln(10)}{20}$$
    $$\alpha_d = \frac{\pi f \sqrt{\varepsilon_r}}{c} \cdot \tan\delta$$

    Which yield the per-unit-length parameters:
    $$R = 2 z_n \alpha_c$$
    $$L = \frac{z_n \sqrt{\varepsilon_r}}{c}$$
    $$G = \frac{2 \alpha_d}{z_n}$$
    $$C = \frac{\sqrt{\varepsilon_r}}{z_n c}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.core import PhysicalLine

        line = PhysicalLine(
            zn=50.0,
            length=1.0,
            epr=2.2,
            A=0.01,
            fA=1.0,
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
    fA : Parameter, default=1.0
        Frequency scaling reference for attenuation in Hz.
    tand : Parameter, default=0.0
        Dielectric loss tangent.
    """
    zn: Parameter = 50.0
    epr: Parameter = 1.0
    A: Parameter = 0.0    
    fA: Parameter = 1.0  
    tand: Parameter = 0.0 

    def rlgc(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        f = freq.f
        sqrt_epr = jnp.sqrt(self.epr)
        A_dB = self.A * jnp.sqrt(f / self.fA)

        alpha_c = A_dB * (jnp.log(10) / 20.0)
        alpha_d = jnp.pi * sqrt_epr * f / c * self.tand
        
        R_val = 2 * self.zn * alpha_c
        L_val = (self.zn * sqrt_epr) / c
        G_val = 2 / self.zn * alpha_d
        C_val = sqrt_epr / (self.zn * c)
        
        ones = jnp.ones(freq.npoints)
        R = R_val * ones
        L = L_val * ones
        G = G_val * ones
        C = C_val * ones
        
        return R, L, G, C    
    

class DatasheetLine(RLGCLine):
    r"""
    Transmission line defined by common datasheet parameters (nominal impedance, 
    dielectric constant, and loss factors). Includes skin effect (`k1`) and 
    dielectric loss (`k2`).

    **Mathematical Formulation**

    The normalized loss coefficients ($k_{1,norm}$, $k_{2,norm}$) depend on `loss_coeffs_normalized`. 
    Attenuation variables scale natively with $\sqrt{\omega}$ and $\omega$:
    $$\alpha_c = k_{1,norm} \cdot \frac{\ln(10)}{20} \cdot \sqrt{\omega}$$
    $$\alpha_d = k_{2,norm} \cdot \frac{\ln(10)}{20} \cdot \omega$$

    Resulting in the per-unit-length components:
    $$R = 2 z_n \alpha_c$$
    $$L = \frac{z_n \sqrt{\varepsilon_r}}{c}$$
    $$G = \frac{2 \alpha_d}{z_n}$$
    $$C = \frac{\sqrt{\varepsilon_r}}{z_n c}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.core import DatasheetLine

        cable = DatasheetLine(
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
    r"""
    Coaxial line defined directly by its physical geometry and material properties. 
    
    **Mathematical Formulation**

    Ideal non-dispersive components ($L'$ and $C'$) and dielectric loss ($G$) are given by:
    $$L' = \frac{\mu_0 \mu_r}{2\pi} \ln\left(\frac{b}{a}\right)$$
    $$C' = \frac{2\pi \varepsilon_0 \varepsilon_r}{\ln(b/a)}$$
    $$G_{diel} = \frac{2\pi \omega \varepsilon_0 \varepsilon_r \tan\delta}{\ln(b/a)}$$

    The internal surface impedance defining frequency-dependent skin resistance ($R_{skin}$) 
    and skin inductance ($L_{skin}$) is governed by:
    $$R_{skin} = \frac{1}{2\pi a} \sqrt{\frac{\omega\mu}{2\sigma_a}} + \frac{1}{2\pi b} \sqrt{\frac{\omega\mu}{2\sigma_b}}$$
    $$L_{skin} = \frac{1}{2\pi a} \sqrt{\frac{\mu}{2\omega\sigma_a}} + \frac{1}{2\pi b} \sqrt{\frac{\mu}{2\omega\sigma_b}}$$

    Where $a$ is the inner radius, $b$ is the outer radius, and $\sigma$ is the conductor conductivity ($1/\rho$).
    The total per-unit-length inductance is $L = L' + L_{skin}$.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.core import CoaxialLine

        phys_cable = CoaxialLine(
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
    r"""
    Microstrip line defined by standard geometric and material properties.
    
    Relies on standard Wheeler approximations. Note that configurations where 
    height > width (h > w) are not yet supported.

    **Mathematical Formulation**

    With ratio $u = \frac{W}{H}$, the effective relative permittivity ($\varepsilon_e$) 
    and ideal impedance terms ($Z_a, Z_e$) are:
    $$\varepsilon_e = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2} \frac{1}{\sqrt{1 + 12/u}}$$
    $$Z_a = \frac{120\pi}{u + 1.393 + 0.667 \ln(u + 1.444)}$$
    $$Z_e = \frac{Z_a}{\sqrt{\varepsilon_e}}$$

    Which provide the per-unit-length components:
    $$L = \frac{Z_e \sqrt{\varepsilon_e}}{c}$$
    $$C = \frac{\sqrt{\varepsilon_e}}{Z_e c}$$
    $$R = \frac{1}{W} \sqrt{2 \mu_0 \rho \omega}$$
    $$G = \frac{1}{Z_a c} \frac{\varepsilon_r (\varepsilon_e - 1)}{\varepsilon_r - 1} \tan\delta \cdot \omega$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.core import MicrostripLine

        phys_microstrip = MicrostripLine(
            w=4e-3,
            h=2.0e-3,
            epr=4.6,
            tand=0.025,
            rho=1.72e-8,
            length=0.5
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s_phys = phys_microstrip.s(freq)    

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
