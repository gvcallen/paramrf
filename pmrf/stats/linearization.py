r"""
The linearisation of a fit at its MAP, and the linearised posterior covariance.

A linearisation is a Gauss–Newton approximation of a fit: the observation Jacobian
$J_h = \partial h / \partial \theta = -\partial r / \partial \theta$ in event space
with respect to the model parameters $\theta$, and the model-parameter Fisher matrix
$F_h = \sum_b J_{h,b}^\top \Sigma_{d,b}^{-1} J_{h,b}$, where the data covariance is
$\Sigma_d = K + \Sigma_n$ per event block. It also carries the Fisher matrix $F_\psi$
of the hyperparameters $\psi$, the evaluator's free parameters, on which
$\Sigma_d$ depends. Build one with
:meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`, and combine one or more with
the prior in :func:`posterior_covariance`; the hyperparameters' covariance is
:func:`hyperparameter_posterior_covariance`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import parax as prx
from jaxtyping import Array, PyTree
from parax.distributions import (
    AbstractDistribution, MultivariateNormalDiag, MultivariateNormalFullCovariance,
    MultivariateNormalTri, Transformed,
)

from pmrf.stats.bijectors import ScalarAffine
from pmrf.stats.discrepancy_models import _group_by_matrix
from pmrf.parameters import (
    Space, _check_space, _joint_prior_paths, _joint_priors, is_param, log_prior,
    tree_param_paths, update, values as _values,
)
from pmrf.utils import field

if TYPE_CHECKING:
    from pmrf.objectives.evaluators import MarginalLogLikelihood


class Linearization(eqx.Module):
    r"""The linearisation of one fit at a model.

    Built by :meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`. Columns of
    :attr:`J_h`, and rows and columns of :attr:`F_h`, follow the model parameters in
    :attr:`theta_names` order, each flattened in C order to its size in
    :attr:`theta_shapes`. Rows and columns of :attr:`F_psi` follow the hyperparameters
    in :attr:`psi_names` order in the same way.
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

    #: The hyperparameter Fisher matrix
    #: $[F_\psi]_{ij} = \tfrac{1}{2} \sum_b \operatorname{tr}\big(\Sigma_{d,b}^{-1}
    #: J_{\Sigma,i}\, \Sigma_{d,b}^{-1} J_{\Sigma,j}\big)$, with
    #: $J_{\Sigma,i} = \partial \Sigma_{d,b} / \partial \psi_i$ and shape
    #: ``(q, q)``. With no free hyperparameters it has shape ``(0, 0)``.
    F_psi: Array

    #: The names of the model parameters, in column order.
    theta_names: tuple[str, ...] = field(static=True)

    #: The shape of each model parameter's value, in :attr:`theta_names` order.
    theta_shapes: tuple[tuple[int, ...], ...] = field(static=True)

    #: The names of the hyperparameters, in :attr:`F_psi` order.
    psi_names: tuple[str, ...] = field(static=True)

    #: The shape of each hyperparameter's value, in :attr:`psi_names` order.
    psi_shapes: tuple[tuple[int, ...], ...] = field(static=True)

    #: The space the parameter values, and so :attr:`J_h`, :attr:`F_h` and
    #: :attr:`F_psi`, are in.
    space: Space = field(static=True)

    def unflatten_theta(self, vector: Array) -> dict[str, Array]:
        """Map a flat vector over the model parameters, such as a row of :attr:`F_h`,
        back to names."""
        return _unflatten(self.theta_names, self.theta_shapes, vector)

    def unflatten_psi(self, vector: Array) -> dict[str, Array]:
        """Map a flat vector over the hyperparameters, such as a row of :attr:`F_psi`,
        back to names."""
        return _unflatten(self.psi_names, self.psi_shapes, vector)


def _unflatten(names, shapes, vector) -> dict[str, Array]:
    sizes = [math.prod(shape) for shape in shapes]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    return {
        name: jnp.reshape(vector[..., offset:offset + size], vector.shape[:-1] + shape)
        for name, shape, offset, size in zip(names, shapes, offsets, sizes)
    }


def _flat(model: PyTree, space: Space, allow_empty: bool = False):
    """The free parameter values of `model` in `space` as one vector, and its map back.

    Returns the names, the shapes, the flat vector, and a function from a flat vector to
    a copy of `model` holding those values. Without free parameters it raises, unless
    `allow_empty`, when the vector is empty.
    """
    _check_space(space)
    named = _values(model, free_only=True, space=space)
    if not named:
        if not allow_empty:
            raise ValueError("The model has no free parameters to linearise over.")
        return (), (), jnp.zeros(0), lambda x: model
    names = tuple(named)
    shapes = tuple(tuple(jnp.shape(value)) for value in named.values())
    vector = jnp.concatenate([jnp.ravel(value) for value in named.values()])

    def rebuild(x):
        return update(model, _unflatten(names, shapes, x), space=space)

    return names, shapes, vector, rebuild


def _linearize(
    residual_fn: Callable[[PyTree], Array],
    model: PyTree,
    covariance_fn: Callable[[PyTree], Array],
    hyperparameters: PyTree,
    space: Space,
) -> Linearization:
    """Linearise a residual function of a model, and its covariance's hyperparameters.

    The evaluator-facing entry point is
    :meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`, which supplies
    `residual_fn`, `covariance_fn`, and itself as `hyperparameters` with any fixed
    tangent basis frozen.

    Parameters
    ----------
    residual_fn : Callable
        Maps a model to its event-space residual, with shape ``(*batch, N)``.
    model : PyTree
        The model to linearise at. Must still be wrapped.
    covariance_fn : Callable
        Maps a copy of `hyperparameters` to $\\Sigma_d$ at `model`, with shape
        ``(*chol_batch, N, N)`` broadcasting to ``batch``. It must not depend on the
        model parameters.
    hyperparameters : PyTree
        The tree whose free parameters are the hyperparameters. Must still be wrapped.
    space : {'declared', 'physical', 'raw'}
        The space of the parameter values differentiated with respect to.

    Returns
    -------
    Linearization
    """
    names, shapes, x0, rebuild = _flat(model, space)
    residual = residual_fn(model)
    J_h = -jax.jacfwd(lambda x: residual_fn(rebuild(x)))(x0)
    chol = jnp.linalg.cholesky(covariance_fn(hyperparameters))

    W = _whitened_jacobian(J_h, chol)
    F_h = jnp.einsum('...nps,...nqs->pq', W, W)

    psi_names, psi_shapes, psi0, rebuild_psi = _flat(hyperparameters, space, allow_empty=True)
    F_psi = _hyperparameter_fisher(
        lambda x: covariance_fn(rebuild_psi(x)), psi0, chol, residual.shape[:-1],
    )
    return Linearization(J_h, residual, chol, F_h, F_psi, names, shapes, psi_names, psi_shapes, space)


def _hyperparameter_fisher(
    covariance_fn: Callable[[Array], Array],
    psi0: Array,
    chol: Array,
    batch_shape: tuple[int, ...],
) -> Array:
    r"""The Fisher matrix of flat hyperparameters $\psi$ through $\Sigma_d(\psi)$.

    With $B_i = L^{-1} J_{\Sigma,i} L^{-\top}$, symmetric as $J_{\Sigma,i}$ is,
    $\operatorname{tr}(\Sigma_d^{-1} J_{\Sigma,i}\, \Sigma_d^{-1} J_{\Sigma,j})
    = \sum B_i \odot B_j$. A factor shared by several event blocks counts once per
    block it covers.
    """
    q = psi0.size
    if q == 0:
        return jnp.zeros((0, 0), chol.dtype)
    J_sigma = jnp.moveaxis(jax.jacfwd(covariance_fn)(psi0), -1, -3)
    L = jnp.broadcast_to(chol[..., None, :, :], J_sigma.shape)
    X = jsp.linalg.solve_triangular(L, J_sigma, lower=True)
    B = jsp.linalg.solve_triangular(L, jnp.swapaxes(X, -1, -2), lower=True)
    blocks = math.prod(batch_shape) // math.prod(chol.shape[:-2])
    return 0.5 * blocks * jnp.einsum('...iab,...jab->ij', B, B)


def _whitened_jacobian(J_h: Array, chol: Array) -> Array:
    """Return $L^{-1}J_h$ grouped by shared covariance factors."""

    # Event blocks sharing a factor are solved together, with the parameter axis
    # carried as one more batch axis the factor is shared along.
    n, p = J_h.shape[-2], J_h.shape[-1]
    matrix_shape, R, _ = _group_by_matrix(chol.shape[:-2], jnp.moveaxis(J_h, -1, 0))
    L = jnp.broadcast_to(chol, chol.shape[:-2] + (n, n)).reshape(matrix_shape + (n, n))
    W = jsp.linalg.solve_triangular(L, R, lower=True)
    return W.reshape(W.shape[:-1] + (p, -1))


_MULTIVARIATE_NORMALS = (MultivariateNormalDiag, MultivariateNormalFullCovariance, MultivariateNormalTri)


def _declared_variance(distribution: AbstractDistribution) -> Array:
    """The variance of a one-dimensional prior, a parameter's own, in declared space."""
    distribution = prx.as_unwrapped(distribution)
    if isinstance(distribution, Transformed):
        # A prior over physical space reaches declared space through a scale.
        bijector = prx.as_unwrapped(distribution.bijector)
        if not isinstance(bijector, ScalarAffine):
            raise NotImplementedError(
                f"A moment-matched prior precision needs an affine map to the prior's space, but "
                f"a prior is mapped through a {type(bijector).__name__}, as a prior over raw space "
                "is. Use prior_precision='hessian'."
            )
        return _declared_variance(distribution.distribution) * bijector.scale ** 2
    try:
        return distribution.variance()
    except NotImplementedError:
        raise NotImplementedError(
            f"A moment-matched prior precision needs the prior's variance(), which a "
            f"{type(distribution).__name__} does not provide. Use prior_precision='hessian'."
        ) from None


def _prior_precision(
    model: PyTree,
    space: Space,
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
    x0: Array,
    rebuild: Callable[[Array], PyTree],
    prior_precision: str,
) -> Array:
    """The prior precision over the flat free parameters of `model` in `space`."""
    log_prior_hessian = lambda: -jax.hessian(lambda x: log_prior(rebuild(x), space=space))(x0)
    if prior_precision == 'hessian':
        return log_prior_hessian()
    if prior_precision != 'moment_matched':
        raise ValueError(
            f"Unknown prior_precision {prior_precision!r}; expected 'moment_matched' or 'hessian'."
        )
    if space == 'raw':
        raise NotImplementedError(
            "A moment-matched prior precision is not available in raw space, whose map to "
            "declared space is nonlinear. Use prior_precision='hessian'."
        )
    for _, joint in _joint_priors(model):
        if joint.space == 'raw':
            raise NotImplementedError(
                "A moment-matched prior precision is not available for a joint prior over raw "
                "space, whose map to declared space is nonlinear. Use prior_precision='hessian'."
            )
        distribution = prx.as_unwrapped(joint.distribution)
        if not isinstance(distribution, _MULTIVARIATE_NORMALS):
            raise NotImplementedError(
                f"A moment-matched prior precision needs a joint prior to be a multivariate "
                f"normal, not a {type(distribution).__name__}. Use prior_precision='hessian'."
            )

    joint_paths = _joint_prior_paths(model)
    resolved = tree_param_paths(model, free_only=True)
    diagonal, in_joint = [], []
    for name, shape in zip(names, shapes):
        path, node = resolved[name]
        in_joint.append(jnp.full(shape, path in joint_paths))
        distribution = node.distribution if is_param(node) and path not in joint_paths else None
        if distribution is None:
            diagonal.append(jnp.zeros(shape, x0.dtype))
            continue
        precision = 1.0 / jnp.broadcast_to(_declared_variance(distribution), shape)
        if space == 'physical':
            precision = precision / node._scale ** 2
        diagonal.append(precision)
    precision = jnp.diag(jnp.concatenate([jnp.ravel(d) for d in diagonal]).astype(x0.dtype))
    in_joint = jnp.concatenate([jnp.ravel(j) for j in in_joint])
    if not joint_paths:
        return precision
    # A multivariate normal is already Gaussian: its block of the curvature is its precision.
    return jnp.where(in_joint[:, None] & in_joint[None, :], log_prior_hessian(), precision)


def posterior_covariance_factor(
    linearizations: Sequence[Linearization],
    model: PyTree,
    space: Space = 'declared',
    prior_precision: Literal['moment_matched', 'hessian'] = 'moment_matched',
) -> Array:
    r"""A square root of the linearised posterior covariance of free parameters.

    **Mathematical Formulation**

    Stack the whitened Jacobians of independent fits with a square root of the
    prior precision, then use the singular values of that stack:

    $$A = \begin{bmatrix}L_1^{-1}J_{h,1} \\ \cdots \\ L_k^{-1}J_{h,k} \\
    \Sigma_0^{-1/2}\end{bmatrix}, \qquad
    A = U S V^\top, \qquad W = V S^{-1}, \qquad
    \Sigma_\text{post} = W W^\top.$$

    With ``prior_precision='moment_matched'``, each prior is replaced by the Gaussian
    of the same mean and variance, so a parameter with its own prior adds

    $$\Sigma_{0,ii}^{-1} = 1 / \operatorname{Var}[\theta_i]$$

    from the distribution's ``variance()``, in `space`: a declared-space variance
    times the squared scale in physical space. A joint multivariate normal prior
    adds its precision block. This keeps a uniform or log-uniform prior from adding
    no information, which would leave a weakly constrained parameter's posterior far
    wider than its prior's support. It applies to the linearisation only; the MAP
    objective keeps the true prior.

    With ``prior_precision='hessian'``, $\Sigma_0^{-1} = -\nabla^2\log p(\theta)$ in
    `space` at the linearisation point. This local approximation is only meaningful
    where the prior is log-concave, and a uniform prior inside its bounds adds zero.

    Either way, a symmetric eigendecomposition of $\Sigma_0^{-1}$ supplies its square
    root, with negative eigenvalues clipped to zero, and a parameter without a prior
    adds zero prior precision.

    Parameters
    ----------
    linearizations : Sequence[Linearization]
        One linearisation per independent fit, each of `model` in `space`.
    model : PyTree
        The model the linearisations were taken at, carrying the prior. Must still be
        wrapped.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space of the parameter values. See :class:`pmrf.Param`.
    prior_precision : {'moment_matched', 'hessian'}, default='moment_matched'
        How the prior's precision is taken, as above.

    Returns
    -------
    jax.Array
        A factor ``W`` with shape ``(p, p)`` such that ``W @ W.T`` is the
        covariance over model parameters in :attr:`~Linearization.theta_names` order.

    Raises
    ------
    ValueError
        If no linearisation is given, one is over other parameters or another space,
        or `prior_precision` is unknown.
    NotImplementedError
        With ``prior_precision='moment_matched'``, if `space` is ``'raw'``, a prior is
        over raw space, a prior has no ``variance()``, or a joint prior is not a
        multivariate normal.
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
    precision = _prior_precision(model, space, names, shapes, x0, rebuild, prior_precision)
    p = x0.size
    factors = [
        jnp.moveaxis(_whitened_jacobian(lin.J_h, lin.chol), -2, -1).reshape(-1, p)
        for lin in linearizations
    ]
    return _covariance_factor(jnp.concatenate([*factors, _psd_root(precision)], axis=0))


def _psd_root(matrix: Array) -> Array:
    r"""A square root $R$ with $R^\top R$ the symmetric part of `matrix`, its negative
    eigenvalues clipped to zero."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    return jnp.sqrt(jnp.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T


def _covariance_factor(A: Array) -> Array:
    r"""$W = V S^{-1}$ from $A = U S V^\top$, so that $W W^\top = (A^\top A)^{-1}$."""
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
    prior_precision: Literal['moment_matched', 'hessian'] = 'moment_matched',
) -> Array:
    r"""The linearised posterior covariance of a model's free parameters.

    See :func:`posterior_covariance_factor` for the factorisation, the choice of
    `prior_precision` and its assumptions. The result has shape ``(p, p)`` in the
    linearisations' :attr:`~Linearization.theta_names` order.
    """
    W = posterior_covariance_factor(linearizations, model, space, prior_precision)
    return W @ W.T


def hyperparameter_posterior_covariance(
    linearization: Linearization,
    evaluator: MarginalLogLikelihood,
    space: Space = 'declared',
    prior_precision: Literal['moment_matched', 'hessian'] = 'moment_matched',
) -> Array:
    r"""The linearised posterior covariance of an evaluator's hyperparameters.

    **Mathematical Formulation**

    The linearised posterior over $[\theta; \psi]$ is block diagonal, since the
    $\theta$–$\psi$ block of the Fisher matrix vanishes (see
    :meth:`pmrf.objectives.evaluators.MarginalLogLikelihood.linearize`). Its
    hyperparameter block is

    $$\Sigma_{\psi,\text{post}} = (F_\psi + \Sigma_{\psi,0}^{-1})^{-1},$$

    with the prior precision $\Sigma_{\psi,0}^{-1}$ taken from the evaluator's
    hyperpriors by the `prior_precision` rule of :func:`posterior_covariance_factor`,
    negative eigenvalues clipped to zero. Unlike $F_h$, $F_\psi$ of different fits does
    not add, since the hyperparameters belong to one evaluator, so this takes one
    linearisation.

    Parameters
    ----------
    linearization : Linearization
        The linearisation of `evaluator`'s fit, in `space`.
    evaluator : MarginalLogLikelihood
        The evaluator linearised, carrying the hyperpriors. Must still be wrapped.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space of the parameter values. See :class:`pmrf.Param`.
    prior_precision : {'moment_matched', 'hessian'}, default='moment_matched'
        How the hyperpriors' precision is taken, as for
        :func:`posterior_covariance_factor`.

    Returns
    -------
    jax.Array
        The covariance with shape ``(q, q)`` over the hyperparameters in
        :attr:`~Linearization.psi_names` order.

    Raises
    ------
    ValueError
        If the linearisation is in another space, has no hyperparameters, or is over
        hyperparameters other than `evaluator`'s, or `prior_precision` is unknown.
    NotImplementedError
        With ``prior_precision='moment_matched'``, as for
        :func:`posterior_covariance_factor`.
    """
    if linearization.space != space:
        raise ValueError(f"The linearisation is in {linearization.space!r} space, not {space!r}.")
    if not linearization.psi_names:
        raise ValueError("The linearisation has no hyperparameters.")
    names, shapes, x0, rebuild = _flat(evaluator, space, allow_empty=True)
    if linearization.psi_names != names or linearization.psi_shapes != shapes:
        raise ValueError(
            "The linearisation is over different hyperparameters from the evaluator: "
            f"{linearization.psi_names} rather than {names}."
        )
    precision = _prior_precision(evaluator, space, names, shapes, x0, rebuild, prior_precision)
    A = jnp.concatenate([_psd_root(linearization.F_psi), _psd_root(precision)], axis=0)
    W = _covariance_factor(A)
    return W @ W.T


__all__ = [
    'Linearization',
    'hyperparameter_posterior_covariance',
    'posterior_covariance',
    'posterior_covariance_factor',
]
