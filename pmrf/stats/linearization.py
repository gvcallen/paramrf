r"""
The linearisation of a fit at its MAP, and the linearised posterior covariance.

A linearisation is a Gauss–Newton approximation of a fit: the observation Jacobian
$J_h = \partial h / \partial \theta = -\partial r / \partial \theta$ in event space
with respect to the model parameters $\theta$, with the discrepancy's hyperparameters
held fixed, and the model-parameter Fisher matrix
$F_h = \sum_b J_{h,b}^\top \Sigma_{d,b}^{-1} J_{h,b}$, where the data covariance is
$\Sigma_d = K + \Sigma_n$ per event block. Build one with
:meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`, and combine one or more with
the prior in :func:`posterior_covariance`.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from pmrf.stats.discrepancy_models import _group_by_matrix
from pmrf.parameters import Space, _check_space, log_prior, update, values as _values
from pmrf.utils import field


class Linearization(eqx.Module):
    r"""The linearisation of one fit at a model.

    Built by :meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`. Columns of
    :attr:`J_h`, and rows and columns of :attr:`F_h`, follow the model parameters in
    :attr:`theta_names` order, each flattened in C order to its size in
    :attr:`theta_shapes`.
    """

    #: The observation Jacobian $J_h = \partial h / \partial \theta
    #: = -\partial r / \partial \theta$ in event space, with shape ``(*batch, N, p)``.
    J_h: Array

    #: The residual $r$ in event space, with shape ``(*batch, N)``.
    residual: Array

    #: The lower Cholesky factor of the data covariance $\Sigma_d = K + \Sigma_n$, at
    #: its own batch shape, which broadcasts to ``batch``.
    chol: Array

    #: The model-parameter Fisher matrix
    #: $F_h = \sum_b J_{h,b}^\top \Sigma_{d,b}^{-1} J_{h,b}$, with shape ``(p, p)``.
    F_h: Array

    #: The names of the model parameters, in column order.
    theta_names: tuple[str, ...] = field(static=True)

    #: The shape of each model parameter's value, in :attr:`theta_names` order.
    theta_shapes: tuple[tuple[int, ...], ...] = field(static=True)

    #: The space the parameter values, and so :attr:`J_h` and :attr:`F_h`, are in.
    space: Space = field(static=True)

    def unflatten_theta(self, vector: Array) -> dict[str, Array]:
        """Map a flat vector over the model parameters, such as a row of :attr:`F_h`,
        back to names."""
        return _unflatten(self.theta_names, self.theta_shapes, vector)


def _unflatten(names, shapes, vector) -> dict[str, Array]:
    sizes = [math.prod(shape) for shape in shapes]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    return {
        name: jnp.reshape(vector[..., offset:offset + size], vector.shape[:-1] + shape)
        for name, shape, offset, size in zip(names, shapes, offsets, sizes)
    }


def _flat(model: PyTree, space: Space):
    """The free parameter values of `model` in `space` as one vector, and its map back.

    Returns the names, the shapes, the flat vector, and a function from a flat vector to
    a copy of `model` holding those values.
    """
    _check_space(space)
    named = _values(model, free_only=True, space=space)
    if not named:
        raise ValueError("The model has no free parameters to linearise over.")
    names = tuple(named)
    shapes = tuple(tuple(jnp.shape(value)) for value in named.values())
    vector = jnp.concatenate([jnp.ravel(value) for value in named.values()])

    def rebuild(x):
        return update(model, _unflatten(names, shapes, x), space=space)

    return names, shapes, vector, rebuild


def _linearize(
    residual_fn: Callable[[PyTree], Array],
    model: PyTree,
    chol: Array,
    space: Space,
) -> Linearization:
    """Linearise a residual function of a model, given the factor of its covariance.

    The evaluator-facing entry point is
    :meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`, which supplies
    `residual_fn` and `chol`.

    Parameters
    ----------
    residual_fn : Callable
        Maps a model to its event-space residual, with shape ``(*batch, N)``.
    model : PyTree
        The model to linearise at. Must still be wrapped.
    chol : jax.Array
        The lower Cholesky factor of $\\Sigma_d$, with shape ``(*chol_batch, N, N)``
        broadcasting to ``batch``.
    space : {'declared', 'physical', 'raw'}
        The space of the parameter values differentiated with respect to.

    Returns
    -------
    Linearization
    """
    names, shapes, x0, rebuild = _flat(model, space)
    residual = residual_fn(model)
    J_h = -jax.jacfwd(lambda x: residual_fn(rebuild(x)))(x0)

    W = _whitened_jacobian(J_h, chol)
    F_h = jnp.einsum('...nps,...nqs->pq', W, W)
    return Linearization(J_h, residual, chol, F_h, names, shapes, space)


def _whitened_jacobian(J_h: Array, chol: Array) -> Array:
    """Return $L^{-1}J_h$ grouped by shared covariance factors."""

    # Event blocks sharing a factor are solved together, with the parameter axis
    # carried as one more batch axis the factor is shared along.
    n, p = J_h.shape[-2], J_h.shape[-1]
    matrix_shape, R, _ = _group_by_matrix(chol.shape[:-2], jnp.moveaxis(J_h, -1, 0))
    L = jnp.broadcast_to(chol, chol.shape[:-2] + (n, n)).reshape(matrix_shape + (n, n))
    W = jsp.linalg.solve_triangular(L, R, lower=True)
    return W.reshape(W.shape[:-1] + (p, -1))


def posterior_covariance_factor(
    linearizations: Sequence[Linearization],
    model: PyTree,
    space: Space = 'declared',
) -> Array:
    r"""A square root of the linearised posterior covariance of free parameters.

    **Mathematical Formulation**

    Stack the whitened Jacobians of independent fits with a square root of the
    prior precision, then use the singular values of that stack:

    $$A = \begin{bmatrix}L_1^{-1}J_{h,1} \\ \cdots \\ L_k^{-1}J_{h,k} \\
    \Sigma_0^{-1/2}\end{bmatrix}, \qquad
    A = U S V^\top, \qquad W = V S^{-1}, \qquad
    \Sigma_\text{post} = W W^\top,$$

    where $\Sigma_0^{-1} = -\nabla^2\log p(\theta)$ in `space`. Its symmetric
    eigendecomposition supplies the square root; negative eigenvalues are clipped to
    zero. This local approximation is only meaningful where the prior is log-concave.
    A parameter without a prior, or with a uniform prior inside its bounds, adds zero
    prior precision.

    Parameters
    ----------
    linearizations : Sequence[Linearization]
        One linearisation per independent fit, each of `model` in `space`.
    model : PyTree
        The model the linearisations were taken at, carrying the prior. Must still be
        wrapped.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space of the parameter values. See :class:`pmrf.Param`.

    Returns
    -------
    jax.Array
        A factor ``W`` with shape ``(p, p)`` such that ``W @ W.T`` is the
        covariance over model parameters in :attr:`~Linearization.theta_names` order.

    Raises
    ------
    ValueError
        If no linearisation is given, or one is over other parameters or another space.
    """
    if not linearizations:
        raise ValueError("At least one linearisation is required.")
    names, shapes, x0, rebuild = _flat(model, space)
    for linearization in linearizations:
        if linearization.space != space:
            raise ValueError(
                f"A linearisation is in {linearization.space!r} space, not {space!r}."
            )
        if linearization.theta_names != names or linearization.theta_shapes != shapes:
            raise ValueError(
                "A linearisation is over different model parameters from the model: "
                f"{linearization.theta_names} rather than {names}."
            )
    prior_precision = -jax.hessian(lambda x: log_prior(rebuild(x), space=space))(x0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (prior_precision + prior_precision.T))
    prior_root = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T
    p = x0.size
    factors = [
        jnp.moveaxis(_whitened_jacobian(lin.J_h, lin.chol), -2, -1).reshape(-1, p)
        for lin in linearizations
    ]
    A = jnp.concatenate([*factors, prior_root], axis=0)
    _, singular_values, vt = jnp.linalg.svd(A, full_matrices=False)
    threshold = singular_values[0] * max(A.shape) * jnp.finfo(A.dtype).eps
    singular_values = eqx.error_if(
        singular_values, singular_values[-1] <= threshold,
        "A parameter direction has neither data nor prior information.",
    )
    V = vt.T
    return V / singular_values


def posterior_covariance(
    linearizations: Sequence[Linearization],
    model: PyTree,
    space: Space = 'declared',
) -> Array:
    r"""The linearised posterior covariance of a model's free parameters.

    See :func:`posterior_covariance_factor` for the factorisation and its
    assumptions. The result has shape ``(p, p)`` in the linearisations'
    :attr:`~Linearization.theta_names` order.
    """
    W = posterior_covariance_factor(linearizations, model, space)
    return W @ W.T


__all__ = [
    'Linearization',
    'posterior_covariance',
    'posterior_covariance_factor',
]
