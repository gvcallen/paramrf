r"""
Modified Bessel functions for complex arguments.

JAX ships :func:`jax.scipy.special.i0e` and :func:`~jax.scipy.special.i1e`
for real arguments only, but cylindrical-conductor physics needs $I_0$ and
$I_1$ at the complex propagation constant $\gamma a=\sqrt{j\omega\mu\sigma}\,a$.
What that physics actually needs is never $I_0$ or $I_1$ alone but their
ratio, which is bounded everywhere off the imaginary axis, so this module
evaluates the ratio directly rather than two exponentially large numbers
that are then divided.
"""
import jax.numpy as jnp

# Number of power-series terms used below the cutoff.
_SERIES_TERMS = 40

#: $|x|$ above which the ratio switches from the power series to the
#: asymptotic expansion. The two agree to 3e-8 relative here, and better on
#: either side; raising it trades series terms for accuracy at the seam.
I0_OVER_I1_SERIES_CUTOFF = 20.0

#: Coefficients of $x^{-n}$, $n=0\ldots5$, in the large-argument expansion of
#: $I_0(x)/I_1(x)$, obtained by dividing the standard asymptotic series
#: $I_\nu(x)\sim e^x(1-(\mu-1)/8x+\ldots)/\sqrt{2\pi x}$ with $\mu=4\nu^2$.
_ASYMPTOTIC_COEFFS = (1.0, 1 / 2, 3 / 8, 3 / 8, 63 / 128, 27 / 32)

#: $|x|$ below which $K_0(x)/K_1(x)$ is evaluated from its convergent
#: logarithmic series. Above this point a ten-term asymptotic expansion is
#: both more accurate and avoids cancellation between the series terms.
K0_OVER_K1_SERIES_CUTOFF = 8.0

_K_RATIO_ASYMPTOTIC_COEFFS = (
    1.0, -1 / 2, 3 / 8, -3 / 8, 63 / 128,
    -27 / 32, 1899 / 1024, -81 / 16, 543483 / 32768, -32427 / 512,
)

_HARMONIC_NUMBERS = [0.0]
for _k in range(1, _SERIES_TERMS):
    _HARMONIC_NUMBERS.append(_HARMONIC_NUMBERS[-1] + 1 / _k)
_HARMONIC_NUMBERS = tuple(_HARMONIC_NUMBERS)


def _i_ratio_series(x):
    """Evaluate $I_0(x)/I_1(x)$ from the ascending power series."""
    k = jnp.arange(_SERIES_TERMS)
    half = x[..., None] / 2
    # (x/2)^{2k}/(k!)^2 as a running product, so no factorial ever overflows.
    step = jnp.where(k == 0, 1.0, (half ** 2) / jnp.where(k == 0, 1.0, k) ** 2)
    terms = jnp.cumprod(step, axis=-1)
    i0 = jnp.sum(terms, axis=-1)
    i1 = jnp.sum(terms * half / (k + 1), axis=-1)
    return i0 / i1


def _i_ratio_asymptotic(x):
    """Evaluate $I_0(x)/I_1(x)$ from the large-argument expansion."""
    inv = 1 / x
    out = jnp.zeros_like(x)
    for c in reversed(_ASYMPTOTIC_COEFFS):
        out = out * inv + c
    return out


def i0_over_i1(x: jnp.ndarray) -> jnp.ndarray:
    r"""
    Ratio $I_0(x)/I_1(x)$ of modified Bessel functions, for complex $x$.

    **Mathematical Formulation**

    Below $|x|=20$ the ascending series is summed directly,
    $$I_0(x)=\sum_{k\ge0}\frac{(x/2)^{2k}}{(k!)^2},\qquad
    I_1(x)=\sum_{k\ge0}\frac{(x/2)^{2k+1}}{k!\,(k+1)!},$$
    to 40 terms; above it the ratio is taken from the large-argument
    expansion
    $$\frac{I_0(x)}{I_1(x)}\sim 1+\frac{1}{2x}+\frac{3}{8x^2}
    +\frac{3}{8x^3}+\frac{63}{128x^4}+\frac{27}{32x^5}.$$
    The small-argument limit $I_0/I_1\to2/x$ falls out of the series with no
    special case, which is what makes the dc limit of a cylindrical
    conductor exact rather than patched.

    **Validity**

    Measured against :func:`scipy.special.ive` over $|x|\in[10^{-6},10^4]$:
    worst relative error 3.1e-8, at the switch point, for $|\arg x|$ up to
    $60^\circ$ -- which covers the $45^\circ$ ray
    $\gamma=\sqrt{j\omega\mu\sigma}$ that a good conductor sits on. The
    large-argument expansion is an asymptotic series about the positive real
    axis and degrades as $\arg x\to90^\circ$ (2e-6 at $70^\circ$, 6e-2 at
    $85^\circ$); this function is not for arguments near the imaginary axis.
    See ``tests/test_math/test_bessel.py`` for the per-regime tolerances.
    Both branches are evaluated on safe arguments, so :func:`jax.grad` is
    finite throughout; the derivative jump at the seam is 1.4e-4 relative.
    $x=0$ is a genuine pole of the ratio and is the caller's to handle.

    Parameters
    ----------
    x : jnp.ndarray
        Argument, real or complex.

    Returns
    -------
    jnp.ndarray
        $I_0(x)/I_1(x)$.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < I0_OVER_I1_SERIES_CUTOFF
    # Each branch is evaluated on an argument the other branch's regime
    # cannot make overflow, so the unused branch never poisons the gradient.
    x_series = jnp.where(small, x, I0_OVER_I1_SERIES_CUTOFF)
    x_asymptotic = jnp.where(small, I0_OVER_I1_SERIES_CUTOFF, x)
    return jnp.where(small, _i_ratio_series(x_series), _i_ratio_asymptotic(x_asymptotic))


def _k_ratio_series(x):
    r"""Evaluate $K_0(x)/K_1(x)$ from the convergent logarithmic series."""
    k = jnp.arange(_SERIES_TERMS)
    half = x[..., None] / 2
    step = jnp.where(k == 0, 1.0, half**2 / jnp.where(k == 0, 1.0, k) ** 2)
    terms = jnp.cumprod(step, axis=-1)
    harmonic = jnp.asarray(_HARMONIC_NUMBERS)

    i0 = jnp.sum(terms, axis=-1)
    i1 = jnp.sum(terms * half / (k + 1), axis=-1)
    log_term = jnp.log(x / 2) + jnp.euler_gamma
    k0 = -log_term * i0 + jnp.sum(terms * harmonic, axis=-1)
    k1 = (
        i0 / x + log_term * i1
        - jnp.sum(terms * (2 * k) * harmonic / x[..., None], axis=-1)
    )
    return k0 / k1


def _k_ratio_asymptotic(x):
    """Evaluate $K_0(x)/K_1(x)$ from its large-argument expansion."""
    inv = 1 / x
    out = jnp.zeros_like(x)
    for coefficient in reversed(_K_RATIO_ASYMPTOTIC_COEFFS):
        out = out * inv + coefficient
    return out


def k0_over_k1(x: jnp.ndarray) -> jnp.ndarray:
    r"""Ratio $K_0(x)/K_1(x)$ of modified Bessel functions for complex $x$.

    **Mathematical Formulation**

    Below $|x|=8$, $K_0$ is evaluated from its logarithmic series and $K_1$
    from $K_1=-K_0'$. Above the switch, the ratio is evaluated directly from
    its ten-term large-argument expansion. This covers the $45^\circ$ ray of
    a good conductor from the weak-skin regime through the half-space limit
    without evaluating exponentially small Bessel functions separately.

    Parameters
    ----------
    x : jnp.ndarray
        Argument, real or complex.

    Returns
    -------
    jnp.ndarray
        $K_0(x)/K_1(x)$.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < K0_OVER_K1_SERIES_CUTOFF
    x_series = jnp.where(small, x, K0_OVER_K1_SERIES_CUTOFF)
    x_asymptotic = jnp.where(small, K0_OVER_K1_SERIES_CUTOFF, x)
    return jnp.where(
        small,
        _k_ratio_series(x_series),
        _k_ratio_asymptotic(x_asymptotic),
    )
