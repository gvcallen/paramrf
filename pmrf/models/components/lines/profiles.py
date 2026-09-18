r"""
Profile shapes: how a parameter varies along a non-uniform line.

A profile is a *shape*, not a line. It maps a normalised position along a line to a
plain physical value, and knows nothing about the line, the parameter it drives, or the
line family. That is what lets one exponential shape drive microstrip width, stripline
width or coaxial inner diameter without being rewritten (ADR-0004).
"""
from __future__ import annotations

from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from jax.scipy.special import i1

from pmrf.constraints import Interval, Positive, RealLine
from pmrf.modules.base import Module
from pmrf.parameters import Param, param

#: Order of the Gauss-Legendre rule used for Klopfenstein's $\phi$. Its integrand is
#: analytic in $y$ on the closed interval, so the rule converges geometrically in the
#: order rather than algebraically. Measured against adaptive quadrature, this order
#: holds the relative error below 1e-15 against tight-tolerance adaptive quadrature for
#: every ripple down to 1e-7, by which point the taper is far longer than anything
#: built; raising it buys nothing.
_QUADRATURE_ORDER = 48

_GL_NODES, _GL_WEIGHTS = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(_QUADRATURE_ORDER))


class AbstractProfile(Module):
    r"""
    Abstract base class for a profile: a shape a parameter follows along a line.

    A profile is a :class:`pmrf.Module` rather than a :class:`pmrf.Model`. It is
    parameter-aware, named and validated like any module, but it has no ports and no
    S-parameters, exactly as materials and the line formulation objects do not.

    Subclasses implement :meth:`evaluate`, which maps a normalised position ``t`` to a
    plain physical value. The value is in the units of whatever the profile is later
    attached to: a profile is never told which parameter it drives, so an exponential
    shape applies unchanged to a microstrip width in metres, to a coaxial inner diameter
    or to a dimensionless relative permittivity. Its coefficients are ordinary
    :func:`pmrf.param` fields, so a plain value and a constrained parameter both work
    with no profile-specific machinery::

        ExponentialProfile(start=2e-3, end=8e-3)
        ExponentialProfile(start=prf.Bounded(2e-3, lower=1e-3, upper=5e-3), end=8e-3)

    **Orientation**

    ``t = 0`` is at port 1 and ``t = 1`` is at port 2. This convention is fixed, and is
    restated wherever a profile is attached. There is no orientation flag: reversing a
    shape swaps its coefficients, because a flag that silently mirrors geometry inside a
    cascade is a debugging hazard.

    **Smoothness**

    Shipped profiles are $C^2$ in ``t`` on the closed interval $[0, 1]$, endpoints
    included. This is a requirement, not an accident. A profiled line samples its
    profiles at section midpoints and builds exact hyperbolic sections, which is the
    exponential midpoint rule, an order-2 Magnus integrator. That integrator is
    time-symmetric only while the sampled function is smooth, and its error then expands
    in *even* powers of the section length, which is what makes Richardson extrapolation
    of the section count $O(h^4)$ rather than $O(h^3)$. A profile with a kink breaks
    that expansion silently. A deliberate kink is expressed by cascading two profiled
    lines, not by a discontinuous profile.

    See Also
    --------
    LinearProfile, ExponentialProfile, KlopfensteinProfile
    """

    @abstractmethod
    def evaluate(self, t: ArrayLike) -> ArrayLike:
        """Evaluate the profile at a normalised position along the line.

        This is **elementwise**: ``t`` may be a scalar or an array of any shape, and the
        result has that same shape. A caller sampling many positions, such as a profiled
        line sampling its section midpoints, passes the whole array in one call.

        Parameters
        ----------
        t : ArrayLike
            Normalised position along the line, in $[0, 1]$. ``t = 0`` is at port 1 and
            ``t = 1`` is at port 2.

        Returns
        -------
        ArrayLike
            The profiled value at ``t``, in the physical units of whatever parameter the
            profile is attached to. Same shape as ``t``.
        """


class LinearProfile(AbstractProfile):
    r"""
    A profile varying linearly between two endpoint values.

    The simplest taper, and the one whose endpoints mean exactly what they say:
    ``evaluate(0) == start`` and ``evaluate(1) == end``.

    **Mathematical Formulation**

    $$p(t) = p_0 + (p_1 - p_0)\,t, \qquad 0 \leq t \leq 1$$

    where $p_0$ is `start` and $p_1$ is `end`. It is $C^\infty$ in $t$, so the
    smoothness requirement on :class:`AbstractProfile` holds trivially.

    Coefficients are unconstrained: a linear interpolation is well defined for any real
    endpoints, and whatever positivity the driven parameter requires is enforced by that
    parameter's own constraint.

    Example
    --------
    .. code-block:: python

        from pmrf.models import LinearProfile

        taper = LinearProfile(start=2e-3, end=8e-3)
        taper.evaluate(0.5)   # 5e-3

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 5.8. Wiley.

    Parameters
    ----------
    start : Param
        Value at ``t = 0``, the port-1 end.
    end : Param
        Value at ``t = 1``, the port-2 end.
    """

    #: Value at ``t = 0``, the port-1 end
    start: Param = param(constraint=RealLine())

    #: Value at ``t = 1``, the port-2 end
    end: Param = param(constraint=RealLine())

    def evaluate(self, t: ArrayLike) -> ArrayLike:
        t = jnp.asarray(t)
        return self.start + (self.end - self.start) * t


class ExponentialProfile(AbstractProfile):
    r"""
    A profile varying exponentially between two endpoint values.

    The classic exponential taper. Its logarithm is linear in $t$, so the midpoint value
    is the *geometric* mean of the endpoints rather than the arithmetic one, and
    ``evaluate(0) == start``, ``evaluate(1) == end``.

    **Mathematical Formulation**

    $$p(t) = p_0 \exp\left(t \ln \frac{p_1}{p_0}\right)
           = p_0 \left(\frac{p_1}{p_0}\right)^{t}, \qquad 0 \leq t \leq 1$$

    where $p_0$ is `start` and $p_1$ is `end`. It is $C^\infty$ in $t$ for positive
    endpoints, so the smoothness requirement on :class:`AbstractProfile` holds.

    Both coefficients are constrained positive: the shape is defined through the ratio
    $p_1/p_0$, which is undefined at zero and non-real for a sign change.

    Example
    --------
    .. code-block:: python

        from pmrf.models import ExponentialProfile

        taper = ExponentialProfile(start=2.0, end=8.0)
        taper.evaluate(0.5)   # 4.0, the geometric mean

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 5.8. Wiley.

    Collin, R. E. (1956). The optimum tapered transmission line matching section.
    Proceedings of the IRE, 44(4), 539-548. doi:10.1109/JRPROC.1956.274938

    Parameters
    ----------
    start : Param
        Value at ``t = 0``, the port-1 end. Must be positive.
    end : Param
        Value at ``t = 1``, the port-2 end. Must be positive.
    """

    #: Value at ``t = 0``, the port-1 end
    start: Param = param(constraint=Positive())

    #: Value at ``t = 1``, the port-2 end
    end: Param = param(constraint=Positive())

    def evaluate(self, t: ArrayLike) -> ArrayLike:
        t = jnp.asarray(t)
        return self.start * jnp.exp(t * jnp.log(self.end / self.start))


class KlopfensteinProfile(AbstractProfile):
    r"""
    Klopfenstein's taper: the shape with the smallest passband reflection for a given
    length.

    .. warning::

        This is a **design** solution, not a validation reference. Klopfenstein derived
        it from the linearised Riccati equation for a non-uniform line, which assumes
        small local reflections and neglects multiple scattering. It defines a shape
        worth simulating; it is emphatically not a reference the simulator is checked
        against. A ParamRF result that disagrees with the design's predicted response is
        not by itself evidence that ParamRF is wrong.

    **Endpoints step**

    Unlike :class:`LinearProfile` and :class:`ExponentialProfile`, this shape does *not*
    reach `start` and `end`. It steps by $\Gamma_m$ at each end:
    $p(0) = p_0 e^{\Gamma_m}$ and $p(1) = p_1 e^{-\Gamma_m}$. Those discontinuities are
    the classic signature of the design, not an implementation artefact, and `start` and
    `end` are the *terminating* values the taper is designed to match, not the values it
    takes.

    **Mathematical Formulation**

    With $p_0$ = `start`, $p_1$ = `end`, $\rho$ = `ripple` and

    $$\Gamma_0 = \tfrac{1}{2}\ln\frac{p_1}{p_0}, \qquad
      \Gamma_m = \rho\,\Gamma_0, \qquad
      A = \operatorname{arccosh}\frac{\Gamma_0}{\Gamma_m} = \operatorname{arccosh}\frac{1}{\rho}$$

    the profile over the body of the taper is

    $$\ln p(t) = \tfrac{1}{2}\ln(p_0 p_1)
        + \frac{\Gamma_0}{\cosh A}\, A^2 \phi(2t - 1, A), \qquad 0 \leq t \leq 1$$

    where $\phi$ is Klopfenstein's odd function

    $$\phi(x, A) = \int_0^{x} \frac{I_1\!\left(A\sqrt{1 - y^2}\right)}{A\sqrt{1 - y^2}}
        \, \mathrm{d}y, \qquad |x| \leq 1$$

    and $I_1$ is the modified Bessel function of the first kind. $\phi$ has no closed
    form and is evaluated here by fixed-order Gauss-Legendre quadrature; the integrand is
    analytic in $y$ on the closed interval, including at $y = \pm 1$ where the ratio
    tends to $\tfrac{1}{2}$, so the rule converges geometrically. The endpoint identity
    $A^2\phi(1, A) = \cosh A - 1$ is what produces the steps described above.

    $\phi$ is odd, so $\ln p$ is antisymmetric about $\tfrac{1}{2}\ln(p_0 p_1)$ and
    $p(t)\,p(1-t) = p_0 p_1$. It is $C^2$ in $t$ on $[0, 1]$, endpoints included: the
    first derivative of $\phi$ is the integrand, which is finite at $x = \pm 1$, and the
    second carries a factor $x$ that cancels the apparent singularity there.

    **Ripple**

    `ripple` is $\Gamma_m/\Gamma_0$: the peak passband reflection as a fraction of the
    reflection of the abrupt step the taper replaces. It is parametrised this way, rather
    than as $\Gamma_m$ directly, because it is then valid on the whole of $(0, 1)$ with
    no condition coupling it to `start` and `end`; an absolute $\Gamma_m$ exceeding
    $\Gamma_0$ makes $A$ complex, which under a fit is a mid-run `nan` rather than a
    rejected value. A design stated as an absolute $\Gamma_m$ converts with
    ``ripple = gamma_m / (0.5 * log(end / start))``.

    The bound $\rho = 1$ is excluded, because there the design degenerates: $A = 0$, the
    correction term vanishes, and the profile is the constant $\sqrt{p_0 p_1}$ -- the
    abrupt step the taper was meant to replace. The quadrature is written to reach that
    limit as a finite value rather than a `nan`, since a *fixed* coefficient is not
    range-checked and can be set there.

    Example
    --------
    .. code-block:: python

        import numpy as np
        from pmrf.models import KlopfensteinProfile

        # Pozar Example 5.8: 50 to 100 ohm with a maximum reflection of 0.02.
        gamma_0 = 0.5 * np.log(100.0 / 50.0)
        taper = KlopfensteinProfile(start=50.0, end=100.0, ripple=0.02 / gamma_0)
        taper.evaluate(0.5)   # 70.71, the geometric mean

    References
    ----------
    Klopfenstein, R. W. (1956). A transmission line taper of improved design.
    Proceedings of the IRE, 44(1), 31-35. doi:10.1109/JRPROC.1956.274847

    Kajfez, D., & Prewitt, J. O. (1973). Correction to "A transmission line taper of
    improved design". IEEE Transactions on Microwave Theory and Techniques, 21(5), 364.
    doi:10.1109/TMTT.1973.1128003

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 5.8. Wiley.

    Parameters
    ----------
    start : Param
        Terminating value at the ``t = 0``, port-1 end. Must be positive.
    end : Param
        Terminating value at the ``t = 1``, port-2 end. Must be positive.
    ripple : Param, default=0.05
        Peak passband reflection as a fraction of $\Gamma_0$, in $(0, 1)$.
    """

    #: Terminating value at the ``t = 0``, port-1 end
    start: Param = param(constraint=Positive())

    #: Terminating value at the ``t = 1``, port-2 end
    end: Param = param(constraint=Positive())

    #: Peak passband reflection as a fraction of the abrupt step's reflection
    ripple: Param = param(default=0.05, constraint=Interval(0.0, 1.0))

    def evaluate(self, t: ArrayLike) -> ArrayLike:
        t = jnp.asarray(t)
        gamma_0 = 0.5 * jnp.log(self.end / self.start)
        a = jnp.arccosh(1.0 / self.ripple)

        log_p = 0.5 * jnp.log(self.start * self.end) + (
            gamma_0 / jnp.cosh(a) * a**2 * _phi(2.0 * t - 1.0, a)
        )
        return jnp.exp(log_p)


def _phi(x: ArrayLike, a: ArrayLike) -> ArrayLike:
    r"""Klopfenstein's $\phi(x, A)$, by Gauss-Legendre quadrature, elementwise in `x`.

    The integration interval $[0, x]$ is mapped onto the rule's $[-1, 1]$, so the whole
    of `x` is handled in one vectorised evaluation with no scan or loop.
    """
    x = jnp.asarray(x)

    # Nodes of the rule, mapped from [-1, 1] onto [0, x] for every element of `x`.
    y = 0.5 * x[..., None] * (_GL_NODES + 1.0)

    z = a * jnp.sqrt(jnp.maximum(1.0 - y**2, 0.0))

    # I1(z)/z, whose limit is 1/2 at z = 0. The guarded denominator keeps both the value
    # and its gradient finite at A = 0 and at the interval endpoints.
    safe_z = jnp.where(z == 0.0, 1.0, z)
    integrand = jnp.where(z == 0.0, 0.5, i1(safe_z) / safe_z)

    return 0.5 * x * jnp.sum(integrand * _GL_WEIGHTS, axis=-1)
