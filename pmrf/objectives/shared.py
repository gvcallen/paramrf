r"""Joint likelihood and predictions for discrepancy shared across datasets.

The fit uses a latent-dimensional Woodbury system, without constructing the
stacked observation covariance.
"""

from math import prod
from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import parax.distributions as dist
from jaxtyping import PyTree

from pmrf.frequency import Frequency
from pmrf.objectives.evaluators import MarginalLogLikelihood
from pmrf.objectives.terms import AbstractTerm
from pmrf.stats.covariance_kernels import cross_gram, gram
from pmrf.stats.likelihoods import GaussianLikelihood
from pmrf.stats.shared_discrepancy import SharedLatent
from pmrf.utils import field, freeze, unwrap_self


class _JointSystem(NamedTuple):
    H: jnp.ndarray
    b: jnp.ndarray
    quadratic: jnp.ndarray
    noise_logdet: jnp.ndarray
    transform_logdet: jnp.ndarray
    L: jnp.ndarray
    C_chol: jnp.ndarray
    transformed_b: jnp.ndarray
    solved: jnp.ndarray
    row_counts: tuple[int, ...]


class SharedMarginalLogLikelihood(AbstractTerm):
    r"""Score datasets whose event-space discrepancies share latent functions.

    The datasets are ``(MarginalLogLikelihood, Frequency)`` pairs, each with no
    local discrepancy. All kernel inputs use the first frequency's unit. Latent
    rows are ordered by latent, then flattened kernel batch index, then frequency.
    A kernel row exists on the union of the fit frequency grids.

    Parameters
    ----------
    datasets : tuple[tuple[MarginalLogLikelihood, Frequency], ...]
        Per-dataset likelihood and fixed fit grid. Each likelihood must have
        Gaussian noise and no local or orthogonal discrepancy.
    latents : tuple[SharedLatent, ...]
        Kernel and mixing pairs shared by all datasets.

    **Mathematical Formulation**
    $$r_l(x) = \sum_j M_{lj}(x)g_j(x) + \varepsilon_l(x),\qquad
    \Sigma = \Phi K_g\Phi^\top + \Sigma_n.$$

    With $K_g=LL^\top$, the evaluated system is
    $C=I+L^\top\Phi^\top\Sigma_n^{-1}\Phi L$. Only latent-sized
    matrices are formed.

    References
    ----------
    ADR-0008 in the ParamRF repository.
    """

    #: Likelihoods and their fixed fit frequencies.
    datasets: tuple[tuple[MarginalLogLikelihood, Frequency], ...]

    #: Latent functions shared across the datasets.
    latents: tuple[SharedLatent, ...]

    #: Sorted union of fit frequencies in the first frequency's unit.
    union_x: jnp.ndarray = field(converter=freeze)

    #: Indices into the union for each dataset's fit points.
    union_indices: tuple[jnp.ndarray, ...]

    def __init__(self, datasets, latents):
        self.datasets = tuple((evaluator, frequency) for evaluator, frequency in datasets)
        self.latents = tuple(latents)
        if not self.datasets or not self.latents:
            raise ValueError("Shared discrepancy requires datasets and latents.")
        for evaluator, frequency in self.datasets:
            if not isinstance(evaluator, MarginalLogLikelihood) or not isinstance(frequency, Frequency):
                raise ValueError("Each dataset must be a (MarginalLogLikelihood, Frequency) pair.")
            if evaluator.discrepancy is not None or evaluator.use_orthogonal_discrepancy:
                raise ValueError("Shared datasets cannot have local or orthogonal discrepancy.")
            if not isinstance(evaluator.likelihood, GaussianLikelihood):
                raise ValueError("Shared datasets require a GaussianLikelihood.")
        multiplier = self.datasets[0][1].multiplier
        union_hz = np.unique(np.concatenate([np.asarray(frequency.f) for _, frequency in self.datasets]))
        self.union_x = jnp.asarray(union_hz / multiplier)
        self.union_indices = tuple(
            jnp.asarray(np.searchsorted(union_hz, np.asarray(frequency.f)), dtype=jnp.int32)
            for _, frequency in self.datasets
        )

    def _mixing(self, prediction, x, blocks, row_counts):
        """Evaluate and validate every latent map on one event-space prediction."""
        n = len(x)
        mixings = []
        for latent, rows in zip(self.latents, row_counts):
            matrix = jnp.asarray(latent.mixing(prediction, x))
            if matrix.shape != (n, blocks, rows):
                raise ValueError(f"Mixing returned shape {matrix.shape}; expected {(n, blocks, rows)}.")
            if jnp.iscomplexobj(matrix):
                raise ValueError("Mixing must return a real event-space matrix.")
            mixings.append(matrix)
        return jnp.concatenate(mixings, axis=-1)

    def _prior_factor(self):
        """Factor each independent kernel row before assembling the latent factor."""
        x = self.union_x
        matrices = []
        row_counts = []
        for latent in self.latents:
            batched = gram(latent.kernel, x, jitter=latent.jitter)
            rows = prod(batched.shape[:-2])
            row_counts.append(rows)
            matrices.extend(jnp.linalg.cholesky(jnp.reshape(batched, (rows, len(x), len(x)))))
        return jsp.linalg.block_diag(*matrices), tuple(row_counts)

    def _statistics(self, model: PyTree, row_counts):
        """Accumulate Phi.T N^-1 Phi, Phi.T N^-1 r and diagonal-noise terms."""
        n_union = len(self.union_x)
        rows = sum(row_counts)
        dimension = rows * n_union
        H = jnp.zeros((dimension, dimension), dtype=self.union_x.dtype)
        b = jnp.zeros((dimension,), dtype=self.union_x.dtype)
        quadratic = jnp.asarray(0.0, dtype=self.union_x.dtype)
        noise_logdet = jnp.asarray(0.0, dtype=self.union_x.dtype)
        transform_logdet = jnp.asarray(0.0, dtype=self.union_x.dtype)
        for (evaluator, frequency), union_index in zip(self.datasets, self.union_indices):
            prediction, residual, variance, logdet = evaluator.event_data(model, frequency)
            n = len(union_index)
            if residual.shape[-1] != n:
                raise ValueError("A dataset's event-space frequency axis must match its fit frequency.")
            blocks = residual.size // n
            residual_nb = residual.reshape((blocks, n)).T
            variance_nb = jnp.broadcast_to(variance, residual.shape).reshape((blocks, n)).T
            x = frequency.f / self.datasets[0][1].multiplier
            mixing = self._mixing(prediction, x, blocks, row_counts)
            precision = 1 / variance_nb
            coordinate = jnp.arange(rows)[None, :] * n_union + union_index[:, None]
            local_H = jnp.einsum('nbi,nbj,nb->nij', mixing, mixing, precision)
            local_b = jnp.einsum('nbi,nb,nb->ni', mixing, residual_nb, precision)
            H = H.at[coordinate[:, :, None], coordinate[:, None, :]].add(local_H)
            b = b.at[coordinate].add(local_b)
            quadratic = quadratic + jnp.sum(residual_nb ** 2 * precision)
            noise_logdet = noise_logdet + jnp.sum(jnp.log(variance_nb))
            transform_logdet = transform_logdet + logdet
        return H, b, quadratic, noise_logdet, transform_logdet

    def _system(self, model):
        L, row_counts = self._prior_factor()
        H, b, quadratic, noise_logdet, transform_logdet = self._statistics(model, row_counts)
        C = jnp.eye(L.shape[0], dtype=L.dtype) + L.T @ H @ L
        C_chol = jnp.linalg.cholesky(C)
        transformed_b = L.T @ b
        solved = jsp.linalg.cho_solve((C_chol, True), transformed_b)
        return _JointSystem(H, b, quadratic, noise_logdet, transform_logdet,
                            L, C_chol, transformed_b, solved, row_counts)

    @unwrap_self
    def __call__(self, model: PyTree) -> jnp.ndarray:
        """Return the joint marginal log-likelihood.

        Parameters
        ----------
        model : PyTree
            Model shared by the dataset predictors.

        Returns
        -------
        jnp.ndarray
            Scalar log density, including conditional-transform Jacobians.
        """
        system = self._system(model)
        n_observations = sum(evaluator.observed.size * (2 if jnp.iscomplexobj(evaluator.observed) else 1)
                             for evaluator, _ in self.datasets)
        logdet = system.noise_logdet + 2 * jnp.sum(jnp.log(jnp.diag(system.C_chol)))
        return -0.5 * (system.quadratic - system.transformed_b @ system.solved + logdet
                       + n_observations * jnp.log(2 * jnp.pi)) + system.transform_logdet

    @unwrap_self
    def predict_latents(self, model: PyTree, new_frequency: Frequency) -> dist.AbstractDistribution:
        """Return one joint Gaussian over latent, row, then new frequency.

        The returned vector excludes measurement noise. Rows follow ``latents``
        order and each kernel's flattened batch order.

        Parameters
        ----------
        model : PyTree
            Model shared by the dataset predictors.
        new_frequency : Frequency
            Grid for the latent posterior, in any unit.

        Returns
        -------
        dist.AbstractDistribution
            Joint Gaussian, flattened by latent, row, then frequency.
        """
        system = self._system(model)
        x_new = new_frequency.f / self.datasets[0][1].multiplier
        cross_blocks = []
        prior_blocks = []
        for latent, rows in zip(self.latents, system.row_counts):
            cross = cross_gram(latent.kernel, x_new, self.union_x).reshape((rows, len(x_new), len(self.union_x)))
            prior = gram(latent.kernel, x_new, jitter=latent.jitter).reshape((rows, len(x_new), len(x_new)))
            cross_blocks.extend(cross)
            prior_blocks.extend(prior)
        K_cross = jsp.linalg.block_diag(*cross_blocks)
        K_new = jsp.linalg.block_diag(*prior_blocks)
        # Q = K_*U L^-T. Triangular solves avoid the cancellation in
        # b - H L C^-1 L.T b when the kernel is smooth on a dense grid.
        Q = jsp.linalg.solve_triangular(system.L, K_cross.T, lower=True).T
        mean = Q @ system.solved
        W = jsp.linalg.solve_triangular(system.C_chol, Q.T, lower=True)
        covariance = K_new - Q @ Q.T + W.T @ W
        covariance = 0.5 * (covariance + covariance.T)
        return dist.MultivariateNormalFullCovariance(mean, covariance)

    @unwrap_self
    def predict_discrepancy(
        self, model: PyTree, new_frequency: Frequency, *,
        dataset: int | None = None, prediction_event: jnp.ndarray | None = None,
    ) -> dist.AbstractDistribution:
        """Predict event-space discrepancy for a fit dataset or a new prediction.

        Pass ``dataset`` for a fitted dataset, or ``prediction_event`` for a new
        device on ``new_frequency``. The returned joint Gaussian is flattened
        in event-block then frequency order.

        Parameters
        ----------
        model : PyTree
            Model shared by the fit dataset predictors.
        new_frequency : Frequency
            Grid for the discrepancy posterior, in any unit.
        dataset : int, optional
            Index of a dataset in the fit. Its predictor supplies the new
            event-space prediction.
        prediction_event : jnp.ndarray, optional
            Event-space prediction for a new device that shares the latents.
            Supply this or ``dataset``, but not both.

        Returns
        -------
        dist.AbstractDistribution
            Joint Gaussian over flattened event blocks and frequency, without
            measurement noise.
        """
        if (dataset is None) == (prediction_event is None):
            raise ValueError("Supply exactly one of dataset or prediction_event.")
        if dataset is not None:
            evaluator, _ = self.datasets[dataset]
            prediction_event = evaluator._event(model, new_frequency)[0]
        latent_prediction = self.predict_latents(model, new_frequency)
        x_new = new_frequency.f / self.datasets[0][1].multiplier
        n = len(x_new)
        if prediction_event.shape[-1] != n:
            raise ValueError("The event-space prediction must match new_frequency.")
        blocks = prediction_event.size // n
        row_counts = tuple(prod(cross_gram(latent.kernel, x_new[:1], x_new[:1]).shape[:-2])
                           for latent in self.latents)
        mixing = self._mixing(prediction_event, x_new, blocks, row_counts)
        rows = sum(row_counts)
        latent_mean = latent_prediction.mean().reshape((rows, n)).T
        mean = jnp.einsum('nbi,ni->bn', mixing, latent_mean).reshape(-1)
        covariance = latent_prediction.covariance().reshape((rows, n, rows, n)).transpose((1, 0, 3, 2))
        covariance = jnp.einsum('nbi,nimj,mcj->bncm', mixing, covariance, mixing)
        covariance = covariance.reshape((blocks * n, blocks * n))
        return dist.MultivariateNormalFullCovariance(mean, covariance)


__all__ = ['SharedMarginalLogLikelihood']
