"""
Discrepancy modeling between an RF model and actual data.

Useful for modeling discrepancy of RF models during fitting.
"""
from collections.abc import Callable
from abc import abstractmethod

import math

import equinox as eqx
import jax
import jax.scipy as jsp
import jax.numpy as jnp
import parax.distributions as dist

from pmrf.covariance_kernels import cross_gram, gram
from pmrf.utils import field
from pmrf.modules.base import Module

class AbstractDiscrepancyModel(Module):
    """
    Abstract base class for discrepancy models.
    
    A discrepancy model maps a model prediction to an updated model prediction.
    This updated prediction can either be deterministic (e.g. a polynomial)
    or probabilistic (e.g. a Gaussian process) by either returning a `JAX` array
    or a `distreqx` probability distribution.
    
    Note that probabilistic discrepancy models operate in "event space".
    Here, probability events (e.g. frequency) are moved to the **last axis**.
    
    These models are commonly used in conjuction with a likelihood function
    via :class:`pmrf.evaluators.MarginalLogLikelihood`.
    
    See :mod:`pmrf.discrepancy_models` for built-in discrepancy models.
    """
    @abstractmethod
    def __call__(self, y_event: jnp.ndarray) -> jnp.ndarray | dist.AbstractDistribution:
        """
        Apply discrepancy correction to a model prediction.

        Parameters
        ----------
        y_event : jnp.ndarray
            The initial model prediction in event space.

        Returns
        -------
        jnp.ndarray | dist.AbstractDistribution
            The updated deterministic or probabilistic prediction.
        """        
        raise NotImplementedError


@jax.custom_vjp
def _gaussian_log_prob(M: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Sum the zero-mean Gaussian log densities of the columns of ``R`` under ``M``.

    ``M`` has shape ``(*batch, N, N)`` and ``R`` has shape ``(*batch, N, k)``: the
    ``k`` residuals in ``R[b]`` share the covariance ``M[b]`` and are solved together.
    """
    return _gaussian_log_prob_fwd(M, R)[0]


def _gaussian_log_prob_fwd(M, R):
    L = jnp.linalg.cholesky(M)
    A = jsp.linalg.cho_solve((L, True), R)
    n, k = R.shape[-2:]
    num_matrices = math.prod(M.shape[:-2])
    logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)))
    value = -0.5 * (
        jnp.sum(R * A) + k * logdet + k * num_matrices * n * jnp.log(2 * jnp.pi)
    )
    return value, (L, A)


def _gaussian_log_prob_bwd(residuals, g):
    # With alpha = M^-1 r, d/dM = (sum alpha alpha^T - k M^-1) / 2 and d/dr = -alpha.
    # JAX has no potri, so M^-1 = L^-T L^-1 from one triangular solve and a matmul.
    L, A = residuals
    n, k = A.shape[-2:]
    identity = jnp.broadcast_to(jnp.eye(n, dtype=L.dtype), L.shape)
    L_inv = jsp.linalg.solve_triangular(L, identity, lower=True)
    M_inv = jnp.swapaxes(L_inv, -1, -2) @ L_inv
    dM = 0.5 * g * (A @ jnp.swapaxes(A, -1, -2) - k * M_inv)
    return dM, -g * A


_gaussian_log_prob.defvjp(_gaussian_log_prob_fwd, _gaussian_log_prob_bwd)


def _add_noise(K: jnp.ndarray, noise_variance, batch_shape) -> jnp.ndarray:
    """Form ``K + sigma^2 I`` at the broadcast batch shape of ``K`` and the noise.

    Raises if that shape does not broadcast to the event ``batch_shape``.
    """
    variance = jnp.asarray(noise_variance)
    M = K + variance[..., None, None] * jnp.eye(K.shape[-1], dtype=K.dtype)
    if jnp.broadcast_shapes(M.shape[:-2], batch_shape) != tuple(batch_shape):
        raise ValueError(
            f"The kernel's batch shape {K.shape[:-2]} and noise variance shape "
            f"{variance.shape} do not broadcast to the event batch shape {tuple(batch_shape)}."
        )
    return M


def _group_by_matrix(matrix_batch_shape, residual):
    """Group the batch entries of ``residual`` by the matrix they share.

    ``matrix_batch_shape`` is the batch shape of a stack of matrices that broadcasts
    to the residual's batch shape ``residual.shape[:-1]``. Batch axes along which the
    matrix is shared become extra right-hand sides. Returns the matrix batch shape
    with the shared axes dropped, the residual reshaped to
    ``(*matrix_shape, N, k)``, and a function that maps an ``(*matrix_shape, M, k)``
    result back to ``(*batch_shape, M)``.
    """
    batch_shape = residual.shape[:-1]
    n = residual.shape[-1]
    padded = (1,) * (len(batch_shape) - len(matrix_batch_shape)) + tuple(matrix_batch_shape)
    matrix_axes = [i for i, size in enumerate(padded) if size == batch_shape[i]]
    shared_axes = [i for i in range(len(batch_shape)) if i not in matrix_axes]
    permutation = matrix_axes + [len(batch_shape)] + shared_axes
    matrix_shape = tuple(batch_shape[i] for i in matrix_axes)
    shared_shape = tuple(batch_shape[i] for i in shared_axes)
    R = jnp.transpose(residual, permutation).reshape(matrix_shape + (n, -1))

    def ungroup(result):
        result = result.reshape(matrix_shape + (result.shape[-2],) + shared_shape)
        return jnp.transpose(result, tuple(sorted(range(len(permutation)), key=permutation.__getitem__)))

    return matrix_shape, R, ungroup


class GaussianProcess(AbstractDiscrepancyModel):
    """
    Gaussian process discrepancy model with a covariance kernel.
    
    Maps model predictions to a Gaussian Process distribution over frequency.
    
    The kernel is responsible for returning the correlation between two input points.
    Given an input `y` of shape `(*batch_shape, event_dims)`, the kernel must accept
    two inputs (x1, x2) of the same shape (scalar or vector), and return an array
    that is broadcastable to `*batch_shape`.
    
    This easily allows for kernel batching. For example, to create multiple RBF kernels
    that model the last batch dimension D with independent kernels, simply create a kernel
    with parameters of shape (D,).
    
    See :class:`pmrf.DiscrepancyModel` for more information on general discrepancy models.
    See :mod:`pmrf.covariance_kernels` for built-in covariance kernels.

    Parameters
    ----------
    kernel : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        The covariance kernel function that computes the correlation between two input arrays.
        Can be a function or a callable PyTree. See :mod:`pmrf.covariance_kernels`
        for built-in covariance kernels.
    jitter : float, default=1e-10
        A small scalar added to the diagonal of the covariance matrix for numerical stability.
    """
    #: The covariance kernel.
    kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    
    #: The added jitter.
    jitter: float = field(default=1e-10, static=True)

    def log_prob(
        self,
        y_event: jnp.ndarray,
        observed: jnp.ndarray,
        x: jnp.ndarray,
        noise_variance: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Evaluate the summed log density of ``observed`` under the GP plus Gaussian noise.

        Each batch entry of ``observed`` is distributed as
        $\mathcal{N}(y, K + \sigma^2 I)$. Equal to the log probability of the
        distribution built by :meth:`__call__` and
        :class:`pmrf.likelihoods.GaussianLikelihood`, summed over the batch.

        ``M = K + sigma^2 I`` is formed and factorized at the broadcast shape of the
        kernel's Gram batch and ``noise_variance``, rather than the full batch shape,
        and the residuals sharing each ``M`` are solved together.

        Parameters
        ----------
        y_event : jnp.ndarray
            The model prediction in event space, with shape ``(*batch_shape, N)``.
        observed : jnp.ndarray
            The observation in event space, broadcastable to ``y_event``.
        x : jnp.ndarray
            The frequency points, with shape ``(N,)``.
        noise_variance : jnp.ndarray
            The noise variance, constant along the event axis and broadcastable to
            ``batch_shape``.

        Returns
        -------
        jnp.ndarray
            The scalar log density, summed over the batch.
        """
        # Materialize K before forming M. Otherwise XLA on CPU fuses the Gram
        # construction into the transpose to LAPACK's column-major layout, and that
        # strided build nearly doubles the value's cost at N = 1000.
        K = jax.lax.optimization_barrier(gram(self.kernel, x, jitter=self.jitter))
        n = y_event.shape[-1]
        M = _add_noise(K, noise_variance, y_event.shape[:-1])
        residual = jnp.broadcast_to(observed - y_event, y_event.shape)
        matrix_shape, R, _ = _group_by_matrix(M.shape[:-2], residual)
        M = M.reshape(matrix_shape + (n, n))
        return _gaussian_log_prob(M, R)

    def predict(
        self,
        residual: jnp.ndarray,
        x: jnp.ndarray,
        x_new: jnp.ndarray,
        noise_variance: jnp.ndarray,
    ) -> dist.AbstractDistribution:
        r"""Predict the discrepancy at new frequencies, given residuals at the fit frequencies.

        Conditions the GP on the residuals $r$ at the fit frequencies $x_A$ with
        Gaussian noise covariance $\Sigma_n = \sigma^2 I$, and returns the
        distribution of the discrepancy $\delta$ at the new frequencies $x_B$ for
        every event block:

        $$\mu = K_{BA} (K_{AA} + \Sigma_n)^{-1} r$$

        $$\Sigma = K_{BB} - K_{BA} (K_{AA} + \Sigma_n)^{-1} K_{AB}$$

        The prediction is of $\delta$ alone: it excludes measurement noise. The
        GP's jitter is added to $K_{AA}$ and $K_{BB}$, so $\Sigma$ stays positive
        definite.

        As in :meth:`log_prob`, ``K_AA + sigma^2 I`` is formed and factorized at the
        broadcast shape of the kernel's Gram batch and ``noise_variance``, and the
        residuals sharing each matrix are solved together.

        Parameters
        ----------
        residual : jnp.ndarray
            The residuals at the fit frequencies in event space, with shape
            ``(*batch_shape, N_A)``.
        x : jnp.ndarray
            The fit frequency points, with shape ``(N_A,)``.
        x_new : jnp.ndarray
            The frequency points to predict at, with shape ``(N_B,)``.
        noise_variance : jnp.ndarray
            The noise variance, constant along the event axis and broadcastable to
            ``batch_shape``.

        Returns
        -------
        dist.AbstractDistribution
            A multivariate normal distribution over the discrepancy at ``x_new``,
            batched over the event blocks, with event shape ``(N_B,)``.
        """
        residual = jnp.asarray(residual)
        batch_shape = residual.shape[:-1]
        M = _add_noise(gram(self.kernel, x, jitter=self.jitter), noise_variance, batch_shape)
        M_batch = M.shape[:-2]
        K_BA = cross_gram(self.kernel, x_new, x)
        K_BB = gram(self.kernel, x_new, jitter=self.jitter)
        n_a, n_b = M.shape[-1], K_BB.shape[-1]
        K_AB = jnp.broadcast_to(jnp.swapaxes(K_BA, -1, -2), M_batch + (n_a, n_b))

        L = jnp.linalg.cholesky(M)
        V = jsp.linalg.solve_triangular(L, K_AB, lower=True)
        covariance = K_BB - jnp.swapaxes(V, -1, -2) @ V

        matrix_shape, R, ungroup = _group_by_matrix(M_batch, residual)
        L = L.reshape(matrix_shape + (n_a, n_a))
        K_AB = K_AB.reshape(matrix_shape + (n_a, n_b))
        mean = ungroup(jnp.swapaxes(K_AB, -1, -2) @ jsp.linalg.cho_solve((L, True), R))

        covariance = jnp.broadcast_to(covariance, batch_shape + (n_b, n_b))
        init_fn = dist.MultivariateNormalFullCovariance
        for _ in batch_shape:
            init_fn = eqx.filter_vmap(init_fn)
        return init_fn(mean, covariance)

    def orthogonal_log_prob(
        self,
        y_event: jnp.ndarray,
        observed: jnp.ndarray,
        x: jnp.ndarray,
        noise_variance: jnp.ndarray,
        basis,
    ) -> jnp.ndarray:
        r"""Evaluate the full-data ``P K P^T + sigma^2 I`` Gaussian density.

        This uses its nonsingular block factorization; it is not REML. In particular,
        the tangent-space block is retained because it depends on the fitted mean and
        measurement noise.
        """
        K = gram(self.kernel, x, jitter=self.jitter)
        variance = jnp.asarray(noise_variance)
        n = y_event.shape[-1]
        # Keep M at the smallest broadcast batch shape. Its Cholesky is then shared
        # automatically across any additional event/basis batch axes.
        M = K + variance[..., None, None] * jnp.eye(n, dtype=K.dtype)
        chol_M = jnp.linalg.cholesky(M)

        def chol_solve(chol, rhs):
            solved = jsp.linalg.solve_triangular(chol, rhs, lower=True)
            return jsp.linalg.solve_triangular(
                jnp.swapaxes(chol, -1, -2), solved, lower=False
            )

        residual = observed - y_event
        Q1 = basis.vectors
        mask = basis.mask
        minv_Q1 = chol_solve(chol_M, Q1)
        minv_r = chol_solve(chol_M, residual[..., None])[..., 0]
        Q1_T = jnp.swapaxes(Q1, -1, -2)
        small = Q1_T @ minv_Q1
        # Rejected SVD columns are padded zeros. Giving those slots an identity block
        # preserves a static shape without contributing to either determinant.
        small = small + (
            jnp.eye(small.shape[-1], dtype=small.dtype)
            * (~mask).astype(small.dtype)[..., None, :]
        )
        chol_small = jnp.linalg.cholesky(small)
        coupling = (Q1_T @ minv_r[..., None])[..., 0]
        corrected = jnp.sum(
            coupling * chol_solve(chol_small, coupling[..., None])[..., 0], axis=-1
        )
        tangent = (Q1_T @ residual[..., None])[..., 0]

        logdet_M = 2 * jnp.sum(
            jnp.log(jnp.diagonal(chol_M, axis1=-2, axis2=-1)), axis=-1
        )
        logdet_small = 2 * jnp.sum(
            jnp.log(jnp.diagonal(chol_small, axis1=-2, axis2=-1)), axis=-1
        )
        rank = jnp.sum(mask, axis=-1)
        quadratic = (
            jnp.sum(tangent**2, axis=-1) / variance
            + jnp.sum(residual * minv_r, axis=-1)
            - corrected
        )
        normalizer = (
            n * jnp.log(2 * jnp.pi)
            + rank * jnp.log(variance)
            + logdet_M
            + logdet_small
        )
        return -0.5 * (normalizer + quadratic)

    def __call__(self, y_event: jnp.ndarray, x: jnp.ndarray, orthogonal_projection: jnp.ndarray | None = None) -> dist.AbstractDistribution:
        """
        Evaluate the Gaussian process distribution over the given inputs.

        Parameters
        ----------
        y_event : jnp.ndarray
            The model prediction in event space, with shape `(..., N)`.
        x : jnp.ndarray
            The frequency points, with shape `(N,)`.
        orthogonal_projection : jnp.ndarray, optional
            An optional matrix P of shape ``(..., N, N)`` which the kernel
            matrix is projected onto using ``P @ K @ P^T``. This can be used
            to specify the subspace which the kernel is allowed.

        Returns
        -------
        dist.AbstractDistribution
            A multivariate normal distribution parameterized by the mean `y_event` 
            and the covariance matrix generated by the kernel.
            
        See Also
        --------
        pmrf.covariance_kernels.gram : Builds the covariance matrix used here.
        """
        K = gram(self.kernel, x, jitter=self.jitter)
        
        if orthogonal_projection is not None:
            projection_T = jnp.swapaxes(orthogonal_projection, -1, -2)
            K = orthogonal_projection @ K @ projection_T

        target_K_shape = y_event.shape[:-1] + K.shape[-2:]
        K = jnp.broadcast_to(K, target_K_shape)

        init_fn = dist.MultivariateNormalFullCovariance
        for _ in range(y_event.ndim - 1):
            init_fn = eqx.filter_vmap(init_fn)
            
        return init_fn(y_event, K)
    
__all__ = [
    'AbstractDiscrepancyModel',
    'GaussianProcess',
]
