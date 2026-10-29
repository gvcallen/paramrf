"""Effective-value transmission-line models and formulations."""
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import c

from pmrf.parameters.constraints import GreaterThan, NonNegative, Positive
from pmrf.frequency import Frequency
from pmrf.materials import DebyePole, DjordjevicSarkarDielectric, LorentzPole, PoleDielectric
from pmrf.models.components.lines.base import AbstractImmittanceLine, AbstractUniformLine, ImmittanceResult
from pmrf.modules.base import Module
from pmrf.parameters import Param, as_param, param
from pmrf.utils import field


class EffectiveValues(eqx.Module):
    r"""
    The effective values that describe an effective-value line.

    Every effective-value line reduces to these five numbers.

    **Mathematical Formulation**

    The attenuation constants are
    $$\alpha_c(\omega) = \frac{\ln 10}{20} A \sqrt{\frac{\omega}{\omega_{ref}}}
    \qquad
    \alpha_d(\omega) = \frac{\omega \sqrt{\varepsilon_{eff}} \tan\delta}{2 c_0}$$
    in Np/m, and the nominal per-unit-length quantities
    $$R'_n = 2 z_n \alpha_c \qquad L'_n = \frac{z_n \sqrt{\varepsilon_{eff}}}{c_0}
    \qquad G'_n = \frac{2 \alpha_d}{z_n} \qquad C'_n = \frac{\sqrt{\varepsilon_{eff}}}{z_n c_0}.$$

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 2.7. Wiley.

    Parameters
    ----------
    zn : ArrayLike
        Nominal characteristic impedance in ohm.
    ep_eff : ArrayLike
        Effective relative permittivity at `f_ref`.
    A : ArrayLike
        Conductor attenuation in dB/m at `f_ref`.
    tand : ArrayLike
        Effective loss tangent at `f_ref`.
    f_ref : ArrayLike
        Reference frequency in Hz.
    """
    #: Nominal characteristic impedance in ohm
    zn: jnp.ndarray

    #: Effective relative permittivity at `f_ref`
    ep_eff: jnp.ndarray

    #: Conductor attenuation in dB/m at `f_ref`
    A: jnp.ndarray

    #: Effective loss tangent at `f_ref`
    tand: jnp.ndarray

    #: Reference frequency in Hz
    f_ref: jnp.ndarray

    def attenuation(self, omega: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""The conductor and dielectric attenuation $(\alpha_c, \alpha_d)$ in Np/m."""
        # sqrt is a branch point at omega = 0. The attenuation is zero there, but
        # its raw gradient is not, so use the double-`where` pattern.
        safe_omega = jnp.where(omega > 0, omega, 1.0)
        w_ref = 2 * jnp.pi * self.f_ref
        root = jnp.where(omega > 0, jnp.sqrt(safe_omega / w_ref), 0.0)
        alpha_c = (jnp.log(10) / 20) * self.A * root
        alpha_d = omega * jnp.sqrt(self.ep_eff) * self.tand / (2 * c)
        return alpha_c, alpha_d

    def rlgc(self, omega: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""The nominal $(R'_n, L'_n, G'_n, C'_n)$, each broadcast over `omega`."""
        alpha_c, alpha_d = self.attenuation(omega)
        sqrt_ep = jnp.sqrt(self.ep_eff)
        ones = jnp.ones(jnp.shape(omega))
        R = 2 * self.zn * alpha_c
        L = self.zn * sqrt_ep / c * ones
        G = 2 * alpha_d / self.zn
        C = sqrt_ep / (self.zn * c) * ones
        return R, L, G, C


def _djordjevic_sarkar(values: EffectiveValues, f_low, f_high) -> DjordjevicSarkarDielectric:
    """The Djordjevic-Sarkar dielectric with the effective values at `f_ref`."""
    # Set the values on a default dielectric rather than passing them to its
    # constructor, which would round-trip them through each field's constraint
    # bijector and give a NaN gradient on a closed bound (ep_eff = 1, tand = 0).
    return eqx.tree_at(
        lambda d: (d.ep_r, d.tand, d.f_low, d.f_high, d.f_ref),
        DjordjevicSarkarDielectric(),
        (values.ep_eff, values.tand, f_low, f_high, values.f_ref),
    )


class AbstractEffectiveLineFormulation(Module):
    """Abstract base class for an effective-line formulation.

    Turns the :class:`EffectiveValues` of an effective-value line into its
    per-unit-length immittance.
    """
    @abstractmethod
    def immittance(self, freq: Frequency, values: EffectiveValues) -> ImmittanceResult:
        """
        Calculate the per-unit-length immittance.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        values : EffectiveValues
            The effective values of the line.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.
        """
        raise NotImplementedError


class NominalEffectiveLineFormulation(AbstractEffectiveLineFormulation):
    r"""
    The RLGC line built directly from the nominal per-unit-length quantities.

    This is the default formulation of :class:`PhysicalLine` and
    :class:`DatasheetLine`, deliberately: CAD "physical" lines, such as ADS/AWR
    ``TLINP`` and scikit-rf's ``DefinedAEpTandZ0`` by default, use a
    frequency-flat permittivity in the same way. It is not causal. $G'$ is
    exactly proportional to $\omega$ with no Kramers-Kronig partner in $C'$, and
    the skin-effect $R'$ has no matching internal inductance. For a causal line,
    use :class:`FirstOrderEffectiveLineFormulation`.

    **Mathematical Formulation**

    $$Z' = R'_n + j\omega L'_n \qquad Y' = G'_n + j\omega C'_n$$

    with the nominal quantities of :class:`EffectiveValues`. To first order in
    $R'/(\omega L')$ and $G'/(\omega C')$ this is :class:`LowLossLine`.

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Sections 2.1 and 2.7. Wiley.
    """
    def immittance(self, freq: Frequency, values: EffectiveValues) -> ImmittanceResult:
        R, L, G, C = values.rlgc(freq.w)
        return ImmittanceResult.from_rlgc(R, L, G, C, freq.w)


class FirstOrderEffectiveLineFormulation(AbstractEffectiveLineFormulation):
    r"""
    The nominal RLGC line with a causal factor on each loss term.

    The conductor loss gains the internal inductance of a half-space conductor,
    and the dielectric becomes a Djordjevic-Sarkar dielectric whose permittivity
    and loss tangent equal the effective values at $f_{ref}$. Both are causal,
    so the line is too. This formulation is opt-in; the default is
    :class:`NominalEffectiveLineFormulation`.

    **Mathematical Formulation**

    $$Z' = (1 + j) R'_n + j\omega L'_n
    \qquad
    Y' = j\omega C'_n \frac{\varepsilon_{DS}(\omega)}{\varepsilon_{eff}}$$

    where $\varepsilon_{DS}$ is the complex permittivity of
    :class:`~pmrf.materials.DjordjevicSarkarDielectric` with
    $\varepsilon_r' = \varepsilon_{eff}$, the line's $\tan\delta$ and $f_{ref}$, and
    this formulation's `f_low` and `f_high`. Equivalently,
    $Y' = \chi_{DS}(\omega) G'_n + j\omega C'_n$ with
    $$X(\omega) = \ln\frac{\omega_{high} + j\omega}{\omega_{low} + j\omega}
    \qquad
    \chi_{DS}(\omega) = \frac{X(\omega) - \Re X(\omega_{ref})}{j \Im X(\omega_{ref})},$$
    so $\chi_{DS}(\omega_{ref}) = 1$ and $Y'$ equals the nominal one at $f_{ref}$.

    References
    ----------
    Djordjevic, A. R., Biljic, R. M., Likar-Smiljanic, V. D., & Sarkar, T. K. (2001).
    Wideband frequency-domain characterization of FR-4 and time-domain causality.
    IEEE Transactions on Electromagnetic Compatibility, 43(4), 662-667.

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.7. Wiley.

    Parameters
    ----------
    f_low : Param, default=1e3
        Lower bound of the relaxation-time distribution in Hz.
    f_high : Param, default=1e12
        Upper bound of the relaxation-time distribution in Hz.
    """
    #: Lower bound of the relaxation-time distribution
    f_low: Param = param(default=1e3, constraint=Positive())

    #: Upper bound of the relaxation-time distribution
    f_high: Param = param(default=1e12, constraint=Positive())

    def immittance(self, freq: Frequency, values: EffectiveValues) -> ImmittanceResult:
        R, L, _, C = values.rlgc(freq.w)
        dielectric = _djordjevic_sarkar(values, self.f_low, self.f_high)
        ep = dielectric.properties(freq).ep_r
        Z = (1 + 1j) * R + 1j * freq.w * L
        Y = 1j * freq.w * C * ep / values.ep_eff
        return ImmittanceResult(Z=Z, Y=Y, omega=freq.w)


class LowLossLine(AbstractUniformLine):
    r"""
    Low-loss line defined by effective values.

    The high-frequency, low-loss approximation of a lossy line. Its
    characteristic impedance is exactly real and its attenuation separates into
    conductor and dielectric parts. It has no per-unit-length immittance, because
    $\sqrt{Z'/Y'}$ is never exactly real once there is loss.

    **Mathematical Formulation**

    $$Z_c(\omega) = z_n
    \qquad
    \gamma(\omega) = \alpha_c(\omega) + \alpha_d(\omega)
    + j\omega\frac{\sqrt{\varepsilon_{eff}}}{c_0}$$

    with
    $$\alpha_c(\omega) = \frac{\ln 10}{20} A \sqrt{\frac{\omega}{\omega_{ref}}}
    \qquad
    \alpha_d(\omega) = \frac{\omega \sqrt{\varepsilon_{eff}} \tan\delta}{2 c_0}.$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import LowLossLine

        line = LowLossLine(zn=50.0, ep_eff=2.1, A=0.05, tand=2e-4, length=1.0)

        freq = prf.Frequency(start=1, stop=1000, npoints=101, unit='mhz')
        s = line.s(freq)

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 2.7. Wiley.

    Parameters
    ----------
    length : Param
        Physical length of the line in meters.
    zn : Param, default=50.0
        Characteristic impedance in ohm.
    ep_eff : Param, default=1.0
        Effective relative permittivity.
    A : Param, default=0.0
        Conductor attenuation in dB/m at `f_ref`.
    tand : Param, default=0.0
        Effective loss tangent.
    f_ref : Param, default=1e9
        Reference frequency of `A` in Hz.
    """
    #: Physical length of the line
    length: Param = param(constraint=Positive())

    #: Characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())

    #: Effective relative permittivity
    ep_eff: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Conductor attenuation in dB/m at `f_ref`
    A: Param = param(default=0.0, constraint=NonNegative())

    #: Effective loss tangent
    tand: Param = param(default=0.0, constraint=NonNegative())

    #: Reference frequency of `A`
    f_ref: Param = param(default=1e9, constraint=Positive())

    def zc_and_gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        values = EffectiveValues(self.zn, self.ep_eff, self.A, self.tand, self.f_ref)
        w = frequency.w
        alpha_c, alpha_d = values.attenuation(w)
        beta = w * jnp.sqrt(self.ep_eff) / c
        gamma = alpha_c + alpha_d + 1j * beta
        zc = self.zn * jnp.ones_like(gamma)
        return zc, gamma * self.length


class AbstractEffectiveLine(AbstractImmittanceLine):
    r"""
    Abstract base class for an effective-value line.

    An effective-value line is parameterized by effective values instead of by
    geometry and materials. Derived classes map their own parameters to
    :class:`EffectiveValues` in :meth:`effective_values`, and turn them into the
    per-unit-length immittance in :meth:`immittance`.

    Parameters
    ----------
    length : Param
        Physical length of the line in meters.
    """
    @abstractmethod
    def effective_values(self) -> EffectiveValues:
        """
        The effective values of the line.

        Returns
        -------
        EffectiveValues
            The nominal impedance, effective permittivity, conductor attenuation,
            loss tangent and their reference frequency.
        """
        raise NotImplementedError


class PhysicalLine(AbstractEffectiveLine):
    r"""
    Effective-value line defined by its effective values directly.

    The default :class:`NominalEffectiveLineFormulation` uses a frequency-flat
    permittivity, as CAD "physical" lines do, and is not causal. Pass
    ``formulation=FirstOrderEffectiveLineFormulation()`` for a causal line.

    **Mathematical Formulation**

    The effective values are the parameters themselves, so
    $$\alpha_c(\omega) = \frac{\ln 10}{20} A \sqrt{\frac{\omega}{\omega_{ref}}}
    \qquad
    \alpha_d(\omega) = \frac{\omega \sqrt{\varepsilon_{eff}} \tan\delta}{2 c_0}$$
    $$R'_n = 2 z_n \alpha_c \qquad L'_n = \frac{z_n \sqrt{\varepsilon_{eff}}}{c_0}
    \qquad G'_n = \frac{2 \alpha_d}{z_n} \qquad C'_n = \frac{\sqrt{\varepsilon_{eff}}}{z_n c_0},$$
    which the formulation turns into $Z'$ and $Y'$.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import PhysicalLine, FirstOrderEffectiveLineFormulation

        line = PhysicalLine(
            zn=50.0,
            length=1.0,
            ep_eff=2.2,
            A=0.3,
            tand=0.001,
            formulation=FirstOrderEffectiveLineFormulation(),
        )

        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        s = line.s(freq)

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Sections 2.1 and 2.7. Wiley.

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance in ohm, defining the L/C ratio.
    ep_eff : Param, default=1.0
        Effective relative permittivity at `f_ref`.
    A : Param, default=0.0
        Conductor attenuation in dB/m at `f_ref`.
    tand : Param, default=0.0
        Effective loss tangent at `f_ref`.
    f_ref : Param, default=1e9
        Reference frequency in Hz of `ep_eff`, `A` and `tand`.
    formulation : AbstractEffectiveLineFormulation, default=NominalEffectiveLineFormulation()
        Turns the effective values into the immittance. The default is not causal;
        see :class:`NominalEffectiveLineFormulation`.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())

    #: Effective relative permittivity at `f_ref`
    ep_eff: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Conductor attenuation in dB/m at `f_ref`
    A: Param = param(default=0.0, constraint=NonNegative())

    #: Effective loss tangent at `f_ref`
    tand: Param = param(default=0.0, constraint=NonNegative())

    #: Reference frequency of `ep_eff`, `A` and `tand`
    f_ref: Param = param(default=1e9, constraint=Positive())

    #: Turns the effective values into the immittance
    formulation: AbstractEffectiveLineFormulation = field(
        default_factory=NominalEffectiveLineFormulation, kw_only=True,
    )

    def effective_values(self) -> EffectiveValues:
        return EffectiveValues(self.zn, self.ep_eff, self.A, self.tand, self.f_ref)

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        return self.formulation.immittance(freq, self.effective_values())


class DatasheetLine(AbstractEffectiveLine):
    r"""
    Effective-value line defined by datasheet velocity factor and loss coefficients.

    The default :class:`NominalEffectiveLineFormulation` uses a frequency-flat
    permittivity, as CAD "physical" lines do, and is not causal. Pass
    ``formulation=FirstOrderEffectiveLineFormulation()`` for a causal line.

    **Mathematical Formulation**

    With $k_1$ in dB/(100 m $\sqrt{\text{MHz}}$) and $k_2$ in dB/(100 m MHz), the
    datasheet attenuation is
    $$\alpha_{dB}(f) = k_1 \sqrt{\frac{f}{1\,\text{MHz}}} + k_2 \frac{f}{1\,\text{MHz}}$$
    per 100 m. Both losses are exact power laws, so they map exactly onto the
    effective values
    $$\varepsilon_{eff} = \frac{1}{v_f^2}
    \qquad
    A = \frac{k_1}{100} \sqrt{\frac{f_{ref}}{1\,\text{MHz}}}
    \qquad
    \tan\delta = \frac{k_2 v_f c_0 \ln 10}{2\pi \cdot 10^9}$$
    at every frequency, not only at $f_{ref}$. :class:`PhysicalLine` gives the
    resulting per-unit-length quantities.

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

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Sections 2.1 and 2.7. Wiley.

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance in ohm.
    vf : Param, default=1.0
        Velocity factor (ratio of propagation speed to the speed of light).
    k1 : Param, default=0.0
        Skin effect loss factor in dB/(100 m sqrt(MHz)).
    k2 : Param, default=0.0
        Dielectric loss factor in dB/(100 m MHz).
    f_ref : Param, default=1e9
        Reference frequency in Hz of the effective values.
    formulation : AbstractEffectiveLineFormulation, default=NominalEffectiveLineFormulation()
        Turns the effective values into the immittance. The default is not causal;
        see :class:`NominalEffectiveLineFormulation`.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())

    #: Velocity factor
    vf: Param = param(default=1.0, constraint=Positive())

    #: Skin effect loss factor in dB/(100 m sqrt(MHz))
    k1: Param = param(default=0.0, constraint=NonNegative())

    #: Dielectric loss factor in dB/(100 m MHz)
    k2: Param = param(default=0.0, constraint=NonNegative())

    #: Reference frequency of the effective values
    f_ref: Param = param(default=1e9, constraint=Positive())

    #: Turns the effective values into the immittance
    formulation: AbstractEffectiveLineFormulation = field(
        default_factory=NominalEffectiveLineFormulation, kw_only=True,
    )

    def effective_values(self) -> EffectiveValues:
        # The datasheet units are per 100 m and per MHz: k1 is k1 / 100 dB/m at 1 MHz,
        # and k2 is k2 / 1e8 dB/m per Hz. tand follows from alpha_d = w sqrt(ep_eff)
        # tand / (2 c0) = (ln 10 / 20) k2 f / 1e8, so its 1e9 is 20 * 1e8 / 2, not f_ref.
        return EffectiveValues(
            zn=self.zn,
            ep_eff=1 / self.vf**2,
            A=self.k1 * jnp.sqrt(self.f_ref / 1e6) / 100,
            tand=self.k2 * self.vf * c * jnp.log(10) / (2 * jnp.pi * 1e9),
            f_ref=self.f_ref,
        )

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        return self.formulation.immittance(freq, self.effective_values())


class RLTerm(Module):
    r"""
    A conductor relaxation of a :class:`GeneralizedLine`.

    A parallel $R_i \parallel L_i$ in series with the line, which adds conductor
    attenuation `A` above `f`.

    **Mathematical Formulation**

    $$Z'_i(\omega) = R_i \frac{j\omega}{\omega_i + j\omega}
    \qquad
    R_i = 2 z_n \frac{\ln 10}{20} A_i
    \qquad
    L_i = \frac{R_i}{\omega_i}$$

    Parameters
    ----------
    A : Param
        Extra conductor attenuation in dB/m contributed above `f`.
    f : Param
        Corner frequency in Hz.
    """
    #: Extra conductor attenuation in dB/m contributed above `f`
    A: Param = param(constraint=NonNegative())

    #: Corner frequency
    f: Param = param(constraint=Positive())


class RCTerm(Module):
    r"""
    A dielectric relaxation of a :class:`GeneralizedLine`.

    A Debye relaxation that adds a loss-tangent peak of height about `tand` at `f`.

    **Mathematical Formulation**

    $$Y'_i(\omega) = j\omega C'_n \frac{2\tan\delta_i}{1 + j\omega/\omega_i}$$

    Parameters
    ----------
    tand : Param
        Height of the loss-tangent peak.
    f : Param
        Relaxation frequency in Hz.
    """
    #: Height of the loss-tangent peak
    tand: Param = param(constraint=NonNegative())

    #: Relaxation frequency
    f: Param = param(constraint=Positive())


class LCTerm(Module):
    r"""
    A modal-dispersion resonance of a :class:`GeneralizedLine`.

    A Lorentz resonance that changes the line's inductance or capacitance by the
    fraction `a` at DC. With $\zeta = 0$ it is a lossless LC term, singular at
    `f`, so `f` must lie outside the evaluated band; with $\zeta > 0$ it is a
    lossy RLC term.

    **Mathematical Formulation**

    $$\Delta_i(\omega) = \frac{a_i}{1 - (\omega/\omega_i)^2 + 2j\zeta_i\omega/\omega_i}
    \overset{\zeta_i = 0}{=} \frac{a_i \omega_i^2}{\omega_i^2 - \omega^2}$$

    Parameters
    ----------
    a : Param
        Fractional change in the line's inductance or capacitance at DC.
    f : Param
        Resonant frequency in Hz.
    zeta : Param, default=0.0
        Dimensionless damping ratio.
    """
    #: Fractional change in the line's inductance or capacitance at DC
    a: Param = param(constraint=NonNegative())

    #: Resonant frequency
    f: Param = param(constraint=Positive())

    #: Dimensionless damping ratio
    zeta: Param = param(default=0.0, constraint=NonNegative())


def _as_terms(cls):
    """A converter that coerces a sequence of terms or plain tuples into `cls` modules."""
    def convert(terms) -> tuple:
        return tuple(term if isinstance(term, cls) else cls(*term) for term in terms)
    return convert


def _debye_pole(dep_r, f_relax) -> DebyePole:
    # Set the values on a default pole, as in `_djordjevic_sarkar`, so that a term
    # on its closed bound (tand = 0) keeps a finite gradient.
    return eqx.tree_at(lambda p: (p.dep_r, p.f_relax), DebyePole(), (dep_r, f_relax))


def _lorentz_pole(dep_r, f_res, zeta) -> LorentzPole:
    return eqx.tree_at(
        lambda p: (p.dep_r, p.f_res, p.zeta), LorentzPole(dep_r=0.0, f_res=1.0), (dep_r, f_res, zeta),
    )


class GeneralizedLine(AbstractEffectiveLine):
    r"""
    Causal effective-value line with RL, RC and LC refinement terms.

    The base is the first-order effective-value line, as
    :class:`PhysicalLine` with :class:`FirstOrderEffectiveLineFormulation`.
    On top of it, an optional skin-effect onset, a DC resistance and four
    families of terms model higher-order behaviour: extra conductor relaxations
    (:class:`RLTerm`), dielectric relaxations (:class:`RCTerm`) and modal
    dispersion of the inductance and capacitance (:class:`LCTerm`). Every part
    is causal, so the line is too. With `R_dc = 0`, `f_s = None` and no terms it
    is exactly the first-order :class:`PhysicalLine`.

    **Mathematical Formulation**

    With the nominal $R'_n(\omega)$, $L'_n$ and $C'_n$ of :class:`EffectiveValues`,
    $$Z'(\omega) = j\omega L'_n \mu(\omega) + Z'_{int}(\omega)
    \qquad
    Y'(\omega) = j\omega C'_n \frac{\varepsilon(\omega)}{\varepsilon_{eff}}.$$

    The medium $(\varepsilon, \mu)$ is a
    :class:`~pmrf.materials.PoleDielectric` whose base is the
    :class:`~pmrf.materials.DjordjevicSarkarDielectric` with
    $\varepsilon_r' = \varepsilon_{eff}$, $\tan\delta$, `f_low`, `f_high` and
    `f_ref`. Each :class:`RCTerm` adds a
    :class:`~pmrf.materials.DebyePole` with
    $\Delta\varepsilon_r = 2\varepsilon_{eff}\tan\delta_i$ and $f_{relax} = f_i$,
    each capacitive :class:`LCTerm` a :class:`~pmrf.materials.LorentzPole` with
    $\Delta\varepsilon_r = \varepsilon_{eff} a_i$, and each inductive
    :class:`LCTerm` a permeability :class:`~pmrf.materials.LorentzPole` with
    $\Delta\mu_r = a_i$, both at $f_{res} = f_i$ with damping $\zeta_i$. So an
    RC term adds
    $$j\omega C'_n \frac{2\tan\delta_i}{1 + j\omega/\omega_i}$$
    to $Y'$, and an undamped LC term adds
    $$j\omega C'_n \frac{a_i \omega_i^2}{\omega_i^2 - \omega^2}
    \quad\text{or}\quad
    j\omega L'_n \frac{a_i \omega_i^2}{\omega_i^2 - \omega^2}$$
    to $Y'$ or $Z'$. A damped one, $\zeta_i > 0$, is a lossy RLC term.

    The conductor's internal impedance is
    $$Z'_{int}(\omega) = R_{dc} + \frac{2}{\pi}(1 + j) R'_n(\omega)
    \arctan\sqrt{\frac{j\omega}{\omega_s}}
    + 2 z_n \frac{\ln 10}{20} \sum_i A_i \frac{j\omega}{\omega_i + j\omega},$$
    where the sum runs over `conductor_terms`, each a parallel
    $R_i \parallel L_i$ with $R_i = 2 z_n \frac{\ln 10}{20} A_i$ and
    $L_i = R_i/\omega_i$. The arctan is the closed form of a distribution of RL
    terms truncated below $\omega_s$,
    $$\frac{\sqrt{2} R'_n(\omega_{ref})}{\pi\sqrt{\omega_{ref}}}
    \int_{\omega_s}^{\infty} \frac{j\omega}{\sqrt{x}(x + j\omega)}\,dx,$$
    which tends to $(1 + j) R'_n(\omega)$ as $\omega_s \to 0$. With
    `f_s = None` the onset is absent and that limit is used exactly.

    `ep_eff` and `tand` describe the *base* medium at `f_ref`. The terms add to
    it, so once any are present $\Re\,\varepsilon(f_{ref}) \neq \varepsilon_{eff}$.
    Each RC term also raises the static capacitance by $2\tan\delta_i C'_n$, as
    the Kramers-Kronig relations require. An undamped LC term is singular at its
    resonance, so `f` must lie outside the evaluated band;
    :class:`~pmrf.materials.LorentzPole` raises if it does not.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import GeneralizedLine, LCTerm

        line = GeneralizedLine(
            zn=50.0,
            ep_eff=2.1,
            A=0.05,
            tand=2e-4,
            length=1.0,
            f_s=1e6,
            conductor_terms=[(0.01, 10e6)],
            dielectric_terms=[(1e-4, 100e6)],
            capacitive_dispersion=[LCTerm(a=1e-3, f=20e9)],
        )

        freq = prf.Frequency(start=1, stop=1000, npoints=101, unit='mhz')
        s = line.s(freq)

    References
    ----------
    Allen, G. V. C. Precision Circuit Modelling for 21 cm Radiometer Calibration.
    PhD thesis.

    Djordjevic, A. R., Biljic, R. M., Likar-Smiljanic, V. D., & Sarkar, T. K. (2001).
    Wideband frequency-domain characterization of FR-4 and time-domain causality.
    IEEE Transactions on Electromagnetic Compatibility, 43(4), 662-667.

    Debye, P. (1929). Polar Molecules. Chemical Catalog Company.

    Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.), Section 7.5. Wiley.

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance in ohm, defining the L/C ratio.
    ep_eff : Param, default=1.0
        Effective relative permittivity of the base medium at `f_ref`.
    A : Param, default=0.0
        Conductor attenuation in dB/m at `f_ref`.
    tand : Param, default=0.0
        Effective loss tangent of the base medium at `f_ref`.
    f_ref : Param, default=1e9
        Reference frequency in Hz of `ep_eff`, `A` and `tand`.
    f_low : Param, default=1e3
        Lower bound of the base medium's relaxation-time distribution in Hz.
    f_high : Param, default=1e12
        Upper bound of the base medium's relaxation-time distribution in Hz.
    R_dc : Param, default=0.0
        DC series resistance in ohm/m.
    f_s : Param | None, default=None
        Skin-effect onset frequency in Hz, or ``None`` for no onset.
    conductor_terms : tuple of RLTerm, default=()
        Conductor relaxations. ``(A, f)`` tuples are coerced to :class:`RLTerm`.
    dielectric_terms : tuple of RCTerm, default=()
        Dielectric relaxations. ``(tand, f)`` tuples are coerced to :class:`RCTerm`.
    inductive_dispersion : tuple of LCTerm, default=()
        Resonances in the inductance. ``(a, f)`` or ``(a, f, zeta)`` tuples are
        coerced to :class:`LCTerm`.
    capacitive_dispersion : tuple of LCTerm, default=()
        Resonances in the capacitance, coerced as `inductive_dispersion`.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())

    #: Effective relative permittivity of the base medium at `f_ref`
    ep_eff: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Conductor attenuation in dB/m at `f_ref`
    A: Param = param(default=0.0, constraint=NonNegative())

    #: Effective loss tangent of the base medium at `f_ref`
    tand: Param = param(default=0.0, constraint=NonNegative())

    #: Reference frequency of `ep_eff`, `A` and `tand`
    f_ref: Param = param(default=1e9, constraint=Positive())

    #: Lower bound of the base medium's relaxation-time distribution
    f_low: Param = param(default=1e3, constraint=Positive())

    #: Upper bound of the base medium's relaxation-time distribution
    f_high: Param = param(default=1e12, constraint=Positive())

    #: DC series resistance in ohm/m
    R_dc: Param = param(default=0.0, constraint=NonNegative())

    #: Skin-effect onset frequency, or None for no onset
    f_s: Param | None = field(
        default=None,
        converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None,
    )

    #: Conductor relaxations
    conductor_terms: tuple[RLTerm, ...] = field(default=(), converter=_as_terms(RLTerm))

    #: Dielectric relaxations
    dielectric_terms: tuple[RCTerm, ...] = field(default=(), converter=_as_terms(RCTerm))

    #: Resonances in the inductance
    inductive_dispersion: tuple[LCTerm, ...] = field(default=(), converter=_as_terms(LCTerm))

    #: Resonances in the capacitance
    capacitive_dispersion: tuple[LCTerm, ...] = field(default=(), converter=_as_terms(LCTerm))

    def effective_values(self) -> EffectiveValues:
        return EffectiveValues(self.zn, self.ep_eff, self.A, self.tand, self.f_ref)

    def _medium(self, values: EffectiveValues) -> PoleDielectric:
        """The medium, built here because `ep_eff` scales the pole strengths."""
        return PoleDielectric(
            _djordjevic_sarkar(values, self.f_low, self.f_high),
            poles=(
                *(_debye_pole(2 * values.ep_eff * t.tand, t.f) for t in self.dielectric_terms),
                *(_lorentz_pole(values.ep_eff * t.a, t.f, t.zeta) for t in self.capacitive_dispersion),
            ),
            mu_poles=tuple(_lorentz_pole(t.a, t.f, t.zeta) for t in self.inductive_dispersion),
        )

    def _internal_impedance(self, w: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
        """The conductor's internal impedance $Z'_{int}$ in ohm/m."""
        skin = (1 + 1j) * R
        if self.f_s is not None:
            # sqrt is a branch point at w = 0, where the onset is zero, so use the
            # double-`where` pattern to keep the gradient finite.
            safe_w = jnp.where(w > 0, w, 1.0)
            root = jnp.where(w > 0, jnp.sqrt(1j * safe_w / (2 * jnp.pi * self.f_s)), 0.0)
            skin = skin * (2 / jnp.pi) * jnp.arctan(root)
        relaxations = sum(
            2 * self.zn * (jnp.log(10) / 20) * t.A * 1j * w / (2 * jnp.pi * t.f + 1j * w)
            for t in self.conductor_terms
        )
        return self.R_dc + skin + relaxations

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        values = self.effective_values()
        w = freq.w
        R, L, _, C = values.rlgc(w)
        medium = self._medium(values).properties(freq)
        Z = 1j * w * L * medium.mu_r + self._internal_impedance(w, R)
        Y = 1j * w * C * medium.ep_r / values.ep_eff
        return ImmittanceResult(Z=Z, Y=Y, omega=w)
