r"""
Modified Bessel functions for complex arguments.

JAX ships :func:`jax.scipy.special.i0e` and :func:`~jax.scipy.special.i1e`
for real arguments only, but cylindrical-conductor physics needs $I_0$ and
$I_1$ at the complex propagation constant $\gamma a=\sqrt{j\omega\mu\sigma}\,a$.
What that physics actually needs is never a Bessel function alone: it is
either a ratio such as $I_0/I_1$, bounded everywhere off the imaginary axis,
or an exponentially scaled function such as $I_1(x)e^{-x}$. Both are
evaluated directly here, so no exponentially large number is ever formed
only to be divided away.
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


def _poly_in_inverse(x, coefficients):
    r"""Evaluate $\sum_k c_k x^{-k}$ by Horner's rule in $1/x$."""
    inv = 1 / x
    out = jnp.zeros_like(x)
    for coefficient in reversed(coefficients):
        out = out * inv + coefficient
    return out


def _i_series_terms(x):
    r"""Return the ascending-series terms of $I_0$ and $I_1$, and their sums.

    Both series share the term $(x/2)^{2k}/(k!)^2$, accumulated as a running
    product so no factorial ever overflows; $I_1$ weights it by
    $(x/2)/(k+1)$. The terms themselves are returned as well because the
    $K$ series below weight the same ones by the harmonic numbers.
    """
    k = jnp.arange(_SERIES_TERMS)
    half = x[..., None] / 2
    step = jnp.where(k == 0, 1.0, (half**2) / jnp.where(k == 0, 1.0, k) ** 2)
    terms = jnp.cumprod(step, axis=-1)
    i0 = jnp.sum(terms, axis=-1)
    i1 = jnp.sum(terms * half / (k + 1), axis=-1)
    return terms, i0, i1


def _k_series(x, terms, i0, i1):
    r"""Return $K_0(x)$ and $K_1(x)$ from their convergent logarithmic series."""
    k = jnp.arange(_SERIES_TERMS)
    harmonic = jnp.asarray(_HARMONIC_NUMBERS)
    log_term = jnp.log(x / 2) + jnp.euler_gamma
    k0 = -log_term * i0 + jnp.sum(terms * harmonic, axis=-1)
    k1 = (
        i0 / x + log_term * i1
        - jnp.sum(terms * (2 * k) * harmonic / x[..., None], axis=-1)
    )
    return k0, k1


def _i_ratio_series(x):
    """Evaluate $I_0(x)/I_1(x)$ from the ascending power series."""
    _, i0, i1 = _i_series_terms(x)
    return i0 / i1


def _i_ratio_asymptotic(x):
    """Evaluate $I_0(x)/I_1(x)$ from the large-argument expansion."""
    return _poly_in_inverse(x, _ASYMPTOTIC_COEFFS)


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
    k0, k1 = _k_series(x, *_i_series_terms(x))
    return k0 / k1


def _k_ratio_asymptotic(x):
    """Evaluate $K_0(x)/K_1(x)$ from its large-argument expansion."""
    return _poly_in_inverse(x, _K_RATIO_ASYMPTOTIC_COEFFS)


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


#: Coefficients $(-1)^k a_k(1)$ of $x^{-k}$ in the large-argument expansion of
#: $\sqrt{2\pi x}\,I_1(x)e^{-x}$, from the standard series
#: $I_\nu(z)\sim e^z(2\pi z)^{-1/2}\sum_k(-1)^k a_k(\nu)z^{-k}$ with
#: $a_k(\nu)=\prod_{m=1}^{k}(4\nu^2-(2m-1)^2)/(k!\,8^k)$ at $\nu=1$.
_I1E_ASYMPTOTIC_COEFFS = (
    1.0, -3 / 8, -15 / 128, -105 / 1024, -4725 / 32768, -72765 / 262144,
)

#: Coefficients $a_k(1)$ of $x^{-k}$ in the large-argument expansion of
#: $\sqrt{2x/\pi}\,K_1(x)e^{x}$, the same $a_k(1)$ as above without the
#: alternating $(-1)^k$, from
#: $K_\nu(z)\sim e^{-z}(\pi/2z)^{1/2}\sum_k a_k(\nu)z^{-k}$. Ten terms are
#: carried rather than six because the logarithmic series it hands over to
#: loses digits to cancellation, so the seam has to sit further out.
_K1E_ASYMPTOTIC_COEFFS = (
    1.0, 3 / 8, -15 / 128, 105 / 1024, -4725 / 32768,
    72765 / 262144, -2837835 / 4194304, 66891825 / 33554432,
    -14783093325 / 2147483648, 468131288625 / 17179869184,
)

#: $|x|$ below which $K_1(x)e^{x}$ is evaluated from its logarithmic series.
#: At the seam on the $45^\circ$ ray the series is 3.5e-8 accurate and the
#: ten-term expansion 1.3e-9; moving it either way makes one of the two
#: worse.
K1E_SERIES_CUTOFF = 12.0


def _i1_series(x):
    r"""Evaluate $I_1(x)$ from the ascending power series."""
    return _i_series_terms(x)[2]


def _k1_series(x):
    r"""Evaluate $K_1(x)$ from its convergent logarithmic series."""
    return _k_series(x, *_i_series_terms(x))[1]


def i1e(x: jnp.ndarray) -> jnp.ndarray:
    r"""
    Exponentially scaled $I_1(x)e^{-x}$, for complex $x$.

    **Mathematical Formulation**

    Below $|x|=20$ the ascending series
    $I_1(x)=\sum_{k\ge0}(x/2)^{2k+1}/(k!\,(k+1)!)$ is summed to 40 terms and
    multiplied by $e^{-x}$; above it the large-argument expansion
    $$I_1(x)e^{-x}\sim\frac{1}{\sqrt{2\pi x}}
    \left(1-\frac{3}{8x}-\frac{15}{128x^2}-\frac{105}{1024x^3}
    -\frac{4725}{32768x^4}-\frac{72765}{262144x^5}\right)$$
    is used instead. Scaling is what keeps a thick-wall tube evaluable:
    $I_1$ itself overflows well before the physics does.

    **Validity**

    Measured against :func:`scipy.special.ive` over $|x|\in[10^{-3},10^4]$
    on the $45^\circ$ ray of a good conductor, worst relative error 1.2e-8
    at the switch point. Like :func:`i0_over_i1`, the large-argument branch is
    an asymptotic series about the positive real axis and degrades as
    $\arg x\to90^\circ$.

    Parameters
    ----------
    x : jnp.ndarray
        Argument, real or complex.

    Returns
    -------
    jnp.ndarray
        $I_1(x)e^{-x}$.

    References
    ----------
    Olver, F. W. J., et al. (eds.). NIST Digital Library of Mathematical
    Functions, 10.25.2, 10.40.1.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < I0_OVER_I1_SERIES_CUTOFF
    x_series = jnp.where(small, x, I0_OVER_I1_SERIES_CUTOFF)
    x_asymptotic = jnp.where(small, I0_OVER_I1_SERIES_CUTOFF, x)
    series = _i1_series(x_series) * jnp.exp(-x_series)
    asymptotic = _poly_in_inverse(x_asymptotic, _I1E_ASYMPTOTIC_COEFFS) / jnp.sqrt(
        2 * jnp.pi * x_asymptotic
    )
    return jnp.where(small, series, asymptotic)


def k1e(x: jnp.ndarray) -> jnp.ndarray:
    r"""
    Exponentially scaled $K_1(x)e^{x}$, for complex $x$.

    **Mathematical Formulation**

    Below $|x|=12$, $K_1$ is evaluated from the logarithmic series used by
    :func:`k0_over_k1` and multiplied by $e^{x}$; above it the ten-term
    large-argument expansion
    $$K_1(x)e^{x}\sim\sqrt{\frac{\pi}{2x}}
    \left(1+\frac{3}{8x}-\frac{15}{128x^2}+\frac{105}{1024x^3}
    -\frac{4725}{32768x^4}+\ldots\right)$$
    is used instead.

    **Validity**

    Measured against :func:`scipy.special.kve` over $|x|\in[10^{-3},10^4]$
    worst relative error 8e-8 on the $45^\circ$ ray of a good conductor, at
    the switch point, rising to 2.5e-7 at $30^\circ$ where the logarithmic
    series suffers more cancellation. Like every asymptotic branch in this module it degrades as
    $\arg x\to90^\circ$. $x=0$ is a genuine pole and is the caller's to
    handle.

    Parameters
    ----------
    x : jnp.ndarray
        Argument, real or complex.

    Returns
    -------
    jnp.ndarray
        $K_1(x)e^{x}$.

    References
    ----------
    Olver, F. W. J., et al. (eds.). NIST Digital Library of Mathematical
    Functions, 10.31.1, 10.40.2.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < K1E_SERIES_CUTOFF
    x_series = jnp.where(small, x, K1E_SERIES_CUTOFF)
    x_asymptotic = jnp.where(small, K1E_SERIES_CUTOFF, x)
    series = _k1_series(x_series) * jnp.exp(x_series)
    asymptotic = _poly_in_inverse(x_asymptotic, _K1E_ASYMPTOTIC_COEFFS) * jnp.sqrt(
        jnp.pi / (2 * x_asymptotic)
    )
    return jnp.where(small, series, asymptotic)
