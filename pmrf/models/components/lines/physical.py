"""
Physical transmission lines (general, coaxial, microstrip)
"""
from scipy.constants import c, mu_0, epsilon_0
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.constraints import Positive, GreaterThan
from pmrf.utils import field
from pmrf.parameters import Param, param, as_param
from pmrf.models.components.lines.base import AbstractRLGCLine, RLGCResult

# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------

class AbstractCoaxialSolver(eqx.Module):
    """Abstract base solver for coaxial line RLGC parameters."""
    def run(self, freq: Frequency, din, dout, epr, mur, tand, rho) -> RLGCResult:
        raise NotImplementedError


class TescheCoaxialSolver(AbstractCoaxialSolver):
    """
    Analytical solver for coaxial line RLGC parameters using the Tesche high-frequency approximation.

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
    
    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy Coaxial 
    Cable Filled With a Nondispersive Dielectric. IEEE Transactions on Electromagnetic 
    Compatibility, 49(1), 12-17.

    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial Transmission Lines 
    and Cylindrical Shields. Bell System Technical Journal, 13(4), 532-579.
    """
    def run(self, freq: Frequency, din, dout, epr, mur, tand, rho) -> RLGCResult:
        eps = epsilon_0 * epr * (1 - 1j * tand)
        mu = mu_0 * mur
        w = freq.w

        a, b = din / 2, dout / 2
        lnbOvera = jnp.log(b/a)
        
        L_prime = jnp.ones(freq.npoints) * mu / (2 * jnp.pi) * lnbOvera
        C_prime = jnp.ones(freq.npoints) * 2 * jnp.pi * jnp.real(eps) / lnbOvera
        G_diel = 2 * jnp.pi * w * -jnp.imag(eps) / lnbOvera
        
        sigma_a, sigma_b = 1 / rho, 1 / rho
        
        L_skin_a = (1 / (2 * jnp.pi * a)) * jnp.sqrt(mu / (2 * w * sigma_a))
        L_skin_b = (1 / (2 * jnp.pi * b)) * jnp.sqrt(mu / (2 * w * sigma_b))
        L_skin = L_skin_a + L_skin_b
                
        R_skin_a = (1 / (2 * jnp.pi * a)) * jnp.sqrt(w * mu / (2 * sigma_a))
        R_skin_b = (1 / (2 * jnp.pi * b)) * jnp.sqrt(w * mu / (2 * sigma_b))
        R_skin = R_skin_a + R_skin_b            
        
        L = L_prime + L_skin
        C = C_prime
        G = G_diel
        R = R_skin
        
        return RLGCResult(R=R, L=L, G=G, C=C)


class AbstractMicrostripSolver(eqx.Module):
    """Abstract base solver for microstrip line RLGC parameters."""
    def run(self, freq: Frequency, w, h, epr, tand, rho, t) -> RLGCResult:
        raise NotImplementedError


class WheelerMicrostripSolver(AbstractMicrostripSolver):
    """
    Standard Wheeler approximations solver for microstrip line RLGC parameters.

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
    
    References
    ----------
    Wheeler, H. A. (1977). Transmission-Line Properties of a Strip on a Dielectric Sheet on a Plane. 
    IEEE Transactions on Microwave Theory and Techniques.
    """
    def run(self, freq: Frequency, w, h, epr, tand, rho, t) -> RLGCResult:
        if t is not None:
            raise ValueError("Wheeler microstrip approximation does not support finite thickness")

        W, H = w, h
        u = W / H

        # Shared base terms
        t1 = (epr + 1) / 2
        t2 = (epr - 1) / 2
        t3 = 1 / jnp.sqrt(1 + 12 / u)

        # Piecewise effective permittivity (epe)
        epe_le1 = t1 + t2 * (t3 + 0.04 * (1 - u)**2)
        epe_gt1 = t1 + t2 * t3
        epe = jnp.where(u <= 1.0, epe_le1, epe_gt1) * jnp.ones(freq.npoints)
        
        # Piecewise characteristic impedance in air (Za)
        Za_le1 = 60 * jnp.log(8 / u + 0.25 * u)
        Za_gt1 = (120 * jnp.pi) / (u + 1.393 + 0.667 * jnp.log(u + 1.444))
        Za = jnp.where(u <= 1.0, Za_le1, Za_gt1)

        # Effective impedance
        Ze = Za / jnp.sqrt(epe)

        # Per-unit-length components
        L = (Ze * jnp.sqrt(epe)) / c
        C = (jnp.sqrt(epe)) / (Ze * c)
        R = (1 / W) * jnp.sqrt(2 * mu_0 * rho) * jnp.sqrt(freq.w)
        G = (1 / (Za * c)) * (epr * (epe - 1) / (epr - 1)) * tand * freq.w
        
        return RLGCResult(R=R, L=L, G=G, C=C)

# -----------------------------------------------------------------------------
# Lines
# -----------------------------------------------------------------------------
    
class PhysicalLine(AbstractRLGCLine):
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
        from pmrf.models import PhysicalLine

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

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance defining the L/C ratio.
    epr : Param, default=1.0
        Relative permittivity.
    A : Param, default=0.0
        Conductor loss in dB/m/sqrt(Hz).
    fA : Param, default=1.0
        Frequency scaling reference for attenuation in Hz.
    tand : Param, default=0.0
        Dielectric loss tangent.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())
    
    #: Relative permittivity
    epr: Param = param(default=1.0, constraint=GreaterThan(1.0))
    
    #: Conductor loss in dB/m/sqrt(Hz)
    A: Param = param(default=0.0, constraint=Positive())
    
    #: Frequency scaling reference
    fA: Param = param(default=1.0, constraint=Positive())
    
    #: Dielectric loss tangent
    tand: Param = param(default=0.0, constraint=Positive())

    def rlgc(self, freq: Frequency) -> RLGCResult:
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
        
        return RLGCResult(R=R, L=L, G=G, C=C)
    

class DatasheetLine(AbstractRLGCLine):
    r"""
    Transmission line defined by common datasheet parameters (nominal impedance
    and velocity/loss factors). Includes skin effect (`k1`) and 
    dielectric loss (`k2`).

    **Mathematical Formulation**

    The normalized loss coefficients ($k_{1,norm}$, $k_{2,norm}$) depend on `loss_coeffs_normalized`. 
    Attenuation variables scale natively with $\sqrt{\omega}$ and $\omega$:
    $$\alpha_c = k_{1,norm} \cdot \frac{\ln(10)}{20} \cdot \sqrt{\omega}$$
    $$\alpha_d = k_{2,norm} \cdot \frac{\ln(10)}{20} \cdot \omega$$

    Resulting in the per-unit-length components:
    $$R = 2 z_n \alpha_c$$
    $$L = \frac{z_n}{v_f c}$$
    $$G = \frac{2 \alpha_d}{z_n}$$
    $$C = \frac{1}{z_n v_f c}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import DatasheetLine

        cable = DatasheetLine(
            zn=50.0,
            vf=0.69,  # Velocity factor (e.g., solid PTFE)
            k1=0.2,   # Skin effect loss factor
            k2=0.01,  # Dielectric loss factor
            length=1.0
        )

        freq = prf.Frequency(start=0.1, stop=10, npoints=201, unit='ghz')
        s = cable.s(freq)

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance.
    vf : Param, default=1.0
        Velocity factor (ratio of propagation speed to the speed of light).
    k1 : Param, default=0.0
        Skin effect loss factor.
    k2 : Param, default=0.0
        Dielectric loss factor.
    loss_coeffs_normalized : bool, default=False
        If True, k1 and k2 are evaluated directly without normalizing to 100MHz references.
    freq_bounds : tuple | None, default=None
        Angular frequency limits (start, stop) used to scale `epr_slope`. Defaults to the analysis array bounds.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())
    
    #: Velocity factor
    vf: Param = param(default=1.0, constraint=Positive())
    
    #: Skin effect loss factor
    k1: Param = param(default=0.0, constraint=Positive())
    
    #: Dielectric loss factor
    k2: Param = param(default=0.0, constraint=Positive())
    
    #: Loss coefficients normalization flag
    loss_coeffs_normalized: bool = field(default=False, static=True)

    def rlgc(self, freq: Frequency) -> RLGCResult:
        w = freq.w
        zn, k1, k2, vf = self.zn, self.k1, self.k2, self.vf

        if not self.loss_coeffs_normalized:
            k1_norm = k1 * (1.0 / (100 * jnp.sqrt(2 * jnp.pi * 10**6)))
            k2_norm = k2 * (1.0 / (100 * 2 * jnp.pi * 10**6))
        else:
            k1_norm = k1
            k2_norm = k2

        sqrt_w = jnp.sqrt(w)
        dBtoNeper = jnp.log(10) / 20
        alpha_c = k1_norm * dBtoNeper * sqrt_w
        alpha_d = k2_norm * dBtoNeper * w
        
        R = 2 * zn * alpha_c
        G = (2 / zn) * alpha_d
        
        # Broadcast L and C to the same shape as frequency arrays 
        # (w) in case zn and vf are provided as scalars.
        L = (zn / (vf * c)) * jnp.ones_like(w)
        C = (1.0 / (zn * vf * c)) * jnp.ones_like(w)
        
        return RLGCResult(R=R, L=L, G=G, C=C)
    

class CoaxialLine(AbstractRLGCLine):
    r"""
    Coaxial line defined directly by its physical geometry and material properties. 
    
    Uses :class:`pmrf.models.TescheCoaxialSolver` as the default mathematical formulation.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import CoaxialLine

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

    Parameters
    ----------
    din : Param, default=1.12e-3
        Inner conductor diameter in meters.
    dout : Param, default=3.2e-3
        Outer conductor inner diameter in meters.
    epr : Param, default=1.0
        Relative permittivity of the dielectric.
    mur : Param, default=1.0
        Relative permeability.
    tand : Param, default=0.0
        Loss tangent of the dielectric.
    rho : Param, default=1.68e-8
        Resistivity of the conductors in Ohm-meters.
    solver : AbstractCoaxialSolver, default=()
        The underlying numerical solver used to compute RLGC parameters. Defaults to TescheCoaxialSolver.
    """
    #: Inner conductor diameter
    din: Param = param(default=1.12e-3, constraint=Positive())
    
    #: Outer conductor inner diameter
    dout: Param = param(default=3.2e-3, constraint=Positive())
    
    #: Relative permittivity
    epr: Param = param(default=1.0, constraint=GreaterThan(1.0))
    
    #: Relative permeability
    mur: Param = param(default=1.0, constraint=Positive())
    
    #: Loss tangent
    tand: Param = param(default=0.0, constraint=Positive())
    
    #: Resistivity of the conductors
    rho: Param = param(default=1.68e-8, constraint=Positive())
    
    #: The underlying physics solver
    solver: AbstractCoaxialSolver = field(default_factory=TescheCoaxialSolver)

    def rlgc(self, freq: Frequency) -> RLGCResult:
        return self.solver.run(
            freq,
            din=self.din,
            dout=self.dout,
            epr=self.epr,
            mur=self.mur,
            tand=self.tand,
            rho=self.rho,
        )
    
    
class MicrostripLine(AbstractRLGCLine):
    r"""
    Microstrip line defined by standard geometric and material properties.
    
    Uses :class:`pmrf.models.WheelerMicrostripSolver` for the default mathematical formulation.
    
    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import MicrostripLine

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

    Parameters
    ----------
    w : Param, default=3e-3
        Width of the microstrip trace in meters.
    h : Param, default=1.6e-3
        Height of the dielectric substrate in meters.
    epr : Param, default=4.3
        Relative permittivity of the dielectric substrate.
    tand : Param, default=0.0
        Dielectric loss tangent.
    rho : Param, default=0.0
        Resistivity of the conductor trace and ground plane in Ohm-meters.
    solver : AbstractMicrostripSolver
        The underlying numerical solver used to compute RLGC parameters. Defaults to WheelerMicrostripSolver.
    """
    #: Width of the microstrip trace
    w: Param = param(default=3e-3, constraint=Positive())
    
    #: Height of the dielectric substrate
    h: Param = param(default=1.6e-3, constraint=Positive())
    
    #: Relative permittivity
    epr: Param = param(default=4.3, constraint=GreaterThan(1.0))
    
    #: Dielectric loss tangent
    tand: Param = param(default=0.0, constraint=Positive())
    
    #: Resistivity of the conductor
    rho: Param = param(default=0.0, constraint=Positive())
    
    #: Thickness of the conductor. Not yet used, provided for future compatibility.
    t: Param | None = field(default=None, converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None)
    
    #: The underlying physics solver
    solver: AbstractMicrostripSolver = field(default_factory=WheelerMicrostripSolver)
    
    def __post_init__(self):
        if self.t is not None:
            raise ValueError("Thickness not yet supported in `MicrostripLine`")

    def rlgc(self, freq: Frequency) -> RLGCResult:
        return self.solver.run(
            freq,
            w=self.w,
            h=self.h,
            epr=self.epr,
            tand=self.tand,
            rho=self.rho,
            t=self.t
        )