"""Log-relative corrections to the internal quantities of a uniform line."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
import parax.distributions as dist

from pmrf.frequency import Frequency
from pmrf.models.adapters.derived import Derived, _unnamed
from pmrf.models.base import Model
from pmrf.models.components.lines.base import AbstractUniformLine
from pmrf.models.port_discrepancy import _values_on_grid
from pmrf.modules.base import Module
from pmrf.parameters import Param, Random, param
from pmrf.stats.distributions import Normal
from pmrf.utils import field, freeze, unwrap, unwrap_self


class AbstractLineDiscrepancy(Module):
    """Real log corrections with shape ``(2, 2, nf)`` on a frequency grid."""

    @abstractmethod
    def __call__(self, frequency: Frequency) -> Array:
        """Evaluate log corrections on the requested grid."""
        raise NotImplementedError


class GridLineDiscrepancy(AbstractLineDiscrepancy):
    r"""Dense line internal discrepancy on one frozen frequency grid.

    ``values[0]`` is real and imaginary log $Z_c$ ratio; ``values[1]`` is
    log attenuation and log phase-constant ratio. A joint prediction in this
    event layout maps directly to ``values``. There is no interpolation.
    """

    #: Real event block × real/imaginary channel × frequency values.
    values: Param = param()
    #: Frozen grid of the values.
    frequency: Frequency = field(converter=freeze)

    def __post_init__(self):
        values = jnp.asarray(unwrap(self.values))
        if jnp.iscomplexobj(values) or values.shape != (2, 2, unwrap(self.frequency).npoints):
            raise ValueError('values must be real with shape (2, 2, nf).')

    @classmethod
    def zeros(cls, frequency: Frequency):
        """Build a zero correction on ``frequency``."""
        return cls(jnp.zeros((2, 2, frequency.npoints)), frequency)

    @classmethod
    def from_prediction(cls, frequency: Frequency, prediction, *, joint: bool = False):
        """Use a discrepancy or joint prediction's mean without block permutation.

        A joint prediction places free parameters first, then the discrepancy
        flattened in C order. The returned carrier uses only that final part;
        retain the original joint distribution when attaching a transfer prior.
        """
        mean = jnp.asarray(prediction.mean())
        shape = (2, 2, frequency.npoints)
        if joint:
            if mean.ndim != 1 or mean.size < 4 * frequency.npoints:
                raise ValueError('joint prediction has no line discrepancy values.')
            values = mean[-4 * frequency.npoints:].reshape(shape)
        else:
            if mean.shape != shape:
                raise ValueError(f'discrepancy prediction must have shape {shape}.')
            values = mean
        return cls(values, frequency)

    @unwrap_self
    def __call__(self, frequency: Frequency) -> Array:
        return _values_on_grid(
            self.values, self.frequency, frequency,
            'Line discrepancy requires its own frequency grid; make a prediction on the new grid.',
        )


class _CorrectedLine(Model):
    #: Unnamed nominal line.
    line: AbstractUniformLine
    #: Internal correction.
    discrepancy: AbstractLineDiscrepancy

    @property
    def number_of_ports(self) -> int:
        return 2

    def s(self, frequency: Frequency, z0=50.0) -> Array:
        """Use the uniform-line scattering equations on corrected internals."""
        return AbstractUniformLine.s(self, frequency, z0)

    def y(self, frequency: Frequency) -> Array:
        """Use the uniform-line admittance equations on corrected internals."""
        return AbstractUniformLine.y(self, frequency)

    def zc_and_gammaL(self, frequency: Frequency) -> tuple[Array, Array]:
        zc, gamma_length = self.line.zc_and_gammaL(frequency)
        delta = self.discrepancy(frequency)
        corrected_zc = zc * jnp.exp(delta[0, 0] + 1j * delta[0, 1])
        corrected_gamma_length = (
            jnp.real(gamma_length) * jnp.exp(delta[1, 0])
            + 1j * jnp.imag(gamma_length) * jnp.exp(delta[1, 1])
        )
        return corrected_zc, corrected_gamma_length


def _correct_line(line, *, discrepancy):
    return _CorrectedLine(line, discrepancy)


class LineCorrected(Derived):
    r"""A lossy uniform line corrected on $Z_c$, $\alpha$ and $\beta$.

    The wrapped line keeps its parameter names; the correction is named
    ``discrepancy.values``. At the root these names are relative, and a circuit
    prefixes both with the component name. The corrected model is itself a
    uniform line, so its S and Y are derived from the corrected internals.

    **Mathematical Formulation**

    $$\check Z_c=Z_c e^{z_r+jz_i},\qquad
      \check\alpha=\alpha e^a,\qquad \check\beta=\beta e^b.$$

    The correction cannot introduce loss when nominal $\alpha=0$. It requires
    positive frequencies with forward phase. Transfer between lengths requires
    the wrapped line's $\gamma L$ to scale with physical length.

    References
    ----------
    Kennedy, M. C. and O'Hagan, A. (2001). Bayesian calibration of computer
    models. Journal of the Royal Statistical Society B, 63(3), 425–464.
    The line-specific log-relative correction is defined in ParamRF issue #283.
    """

    def __init__(self, line: AbstractUniformLine, discrepancy: AbstractLineDiscrepancy, *, name=None, metadata=None):
        super().__init__(
            (_unnamed(line), {'discrepancy': _unnamed(discrepancy)}),
            fn=_correct_line, name=line.name if name is None else name, metadata=metadata,
        )


def reference_line_features(zc: Array, gamma: Array) -> Array:
    r"""Transform reference $(Z_c,\gamma)$ into the carrier's two complex features.

    ``gamma`` is per unit length. Both attenuation and forward phase constant
    must be positive. Measurement-noise variance for a KOH fit using these
    features is specified in the transformed ``(2, 2, nf)`` event space.
    """
    zc, gamma = jnp.asarray(zc), jnp.asarray(gamma)
    if zc.shape != gamma.shape or zc.ndim != 1:
        raise ValueError('zc and gamma must have matching one-dimensional shapes.')
    alpha = jnp.real(gamma)
    beta = jnp.imag(gamma)
    return jnp.stack((jnp.log(zc), jnp.log(alpha) + 1j * jnp.log(beta)), axis=-1)


def line_internal_features(line: AbstractUniformLine, frequency: Frequency) -> Array:
    """Predict the same log features from a uniform line on ``frequency``."""
    zc, gamma_length = line.zc_and_gammaL(frequency)
    return reference_line_features(zc, gamma_length / unwrap(line.length))


class BasisLineDiscrepancy(AbstractLineDiscrepancy):
    r"""Four-channel Hilbert-space GP basis with standard-normal coefficients.

    The kernel and its spectral density are supplied with fixed hyperparameters
    at construction. Dirichlet Laplacian eigenfunctions on ``domain`` are scaled
    by the square root of the spectral density; the smallest rank meeting the
    requested prior-variance tolerance on the reference grid is retained.

    **Mathematical Formulation**

    $$\delta_k(f)=\sum_{j=1}^m u_{kj}\sqrt{S(\omega_j)}
      \sqrt{2/(b-a)}\sin[\omega_j(f-a)],\quad
      \omega_j=j\pi/(b-a),\quad u_{kj}\sim N(0,1).$$

    References
    ----------
    Solin, A. and Särkkä, S. (2020). Hilbert space methods for reduced-rank
    Gaussian process regression. Statistics and Computing 30, 419–446.
    """

    #: Four channels of standard-normal basis coefficients, shape (2, 2, rank).
    coefficients: Param = param()
    #: Allowed frequency interval in Hz.
    domain: tuple[float, float] = field(static=True, converter=tuple)
    #: Square roots of spectral densities at the selected frequencies.
    spectral_weights: Array = field(converter=freeze)

    @classmethod
    def from_kernel(
        cls, kernel, spectral_density, domain: tuple[float, float],
        reference_frequency: Frequency, *, variance_tolerance: float = 0.05,
        max_rank: int = 64,
    ):
        """Select the first rank meeting relative prior-variance error on the grid.

        ``spectral_density(omega)`` uses angular frequency dual to Hz. The
        supplied kernel supplies its zero-lag variance, while its spectral
        density supplies basis weights; both must describe the same kernel.
        """
        low, high = map(float, domain)
        if not low < high or max_rank < 1 or not 0 < variance_tolerance < 1:
            raise ValueError('Require an increasing domain, positive rank and tolerance in (0, 1).')
        f = np.asarray(reference_frequency.f)
        if np.any(f < low) or np.any(f > high):
            raise ValueError('Reference grid is outside the basis domain.')
        indices = np.arange(1, max_rank + 1)
        omega = indices * np.pi / (high - low)
        spectrum = np.asarray(spectral_density(jnp.asarray(omega)))
        if spectrum.shape != omega.shape or np.any(spectrum < 0) or not np.all(np.isfinite(spectrum)):
            raise ValueError('Spectral density must be finite and non-negative at each eigenfrequency.')
        prior_variance = float(np.asarray(kernel(jnp.asarray([0.0]), jnp.asarray([0.0]))))
        if prior_variance <= 0:
            raise ValueError('Kernel prior variance must be positive.')
        basis = np.sqrt(2 / (high - low) * spectrum)[:, None] * np.sin(omega[:, None] * (f - low))
        variance = np.cumsum(basis**2, axis=0)
        errors = np.max(np.abs(variance / prior_variance - 1), axis=1)
        candidates = np.flatnonzero(errors <= variance_tolerance)
        if len(candidates) == 0:
            raise ValueError('No rank meets the prior-variance tolerance on the reference grid.')
        rank = int(candidates[0]) + 1
        coefficients = Random(Normal(0.0, 1.0), value=jnp.zeros((2, 2, rank)))
        return cls(coefficients, (low, high), jnp.asarray(np.sqrt(spectrum[:rank])))

    def basis(self, frequency: Frequency) -> Array:
        """Evaluate the spectral basis on any grid inside the domain."""
        low, high = self.domain
        f = frequency.f
        out = jnp.logical_or(jnp.any(f < low), jnp.any(f > high))
        weights = eqx.error_if(unwrap(self.spectral_weights), out,
                               'Frequency is outside the line discrepancy basis domain.')
        omega = jnp.arange(1, weights.shape[0] + 1) * jnp.pi / (high - low)
        return (jnp.sqrt(2 / (high - low)) * weights[:, None]
                * jnp.sin(omega[:, None] * (f - low))).T

    @unwrap_self
    def __call__(self, frequency: Frequency) -> Array:
        return jnp.einsum('f j, a b j -> a b f', self.basis(frequency), self.coefficients)

    def materialize(self, frequency: Frequency) -> GridLineDiscrepancy:
        """Evaluate coefficients as a dense carrier on the requested grid."""
        return GridLineDiscrepancy(self(frequency), frequency)


def project_line_basis_joint(model, basis_discrepancy: BasisLineDiscrepancy,
                             linearization, covariance, frequency: Frequency, *,
                             discrepancy_name: str = 'discrepancy.coefficients', jitter: float = 1e-12):
    """Project a Laplace fit over coefficients and line parameters to carrier values.

    Returns names and a joint Gaussian ordered as the fitted free parameters
    other than coefficients, followed by the dense carrier values. The full
    covariance, including cross-covariance, is retained. ``jitter`` gives the
    rank-deficient dense carrier a proper density for prior attachment.
    """
    from pmrf.parameters import values

    names = linearization.names
    if discrepancy_name not in names:
        raise ValueError(f'{discrepancy_name!r} is not in the linearisation.')
    coeff_index = names.index(discrepancy_name)
    sizes = [int(np.prod(shape)) for shape in linearization.shapes]
    offsets = np.cumsum([0, *sizes])
    p = offsets[-1]
    covariance = jnp.asarray(covariance)
    if covariance.shape != (p, p):
        raise ValueError(f'Covariance must have shape {(p, p)}.')
    coeff = dict(values(model, free_only=True, space='declared'))[discrepancy_name]
    rank = coeff.shape[-1]
    if coeff.shape != (2, 2, rank):
        raise ValueError('Basis coefficients must have shape (2, 2, rank).')
    basis = basis_discrepancy.basis(frequency)
    projection = jnp.zeros((4 * frequency.npoints, p))
    for channel in range(4):
        row = slice(channel * frequency.npoints, (channel + 1) * frequency.npoints)
        col = slice(offsets[coeff_index] + channel * rank, offsets[coeff_index] + (channel + 1) * rank)
        projection = projection.at[row, col].set(basis)
    keep = np.r_[0:offsets[coeff_index], offsets[coeff_index + 1]:p]
    selector = jnp.eye(p)[keep]
    transform = jnp.concatenate((selector, projection), axis=0)
    flat = jnp.concatenate([jnp.ravel(value) for value in values(model, free_only=True, space='declared').values()])
    mean = transform @ flat
    joint_covariance = transform @ covariance @ transform.T
    joint_covariance = joint_covariance.at[len(keep):, len(keep):].add(jnp.eye(4 * frequency.npoints) * jitter)
    return tuple(name for name in names if name != discrepancy_name), dist.MultivariateNormalFullCovariance(mean, joint_covariance)


__all__ = [
    'AbstractLineDiscrepancy', 'GridLineDiscrepancy', 'LineCorrected',
    'reference_line_features', 'line_internal_features',
    'BasisLineDiscrepancy', 'project_line_basis_joint',
]
