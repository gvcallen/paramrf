import jax
import jax.numpy as jnp
from jax import vmap, random

from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model

class KrigingUncertaintySampler(AdaptiveSampler):
    def __init__(
        self,
        model: Model,
        variance_threshold=0.01,
        initial_models: list[Model] | int = 10,
        *args,
        **kwargs
    ):
        self.variance_threshold = variance_threshold
        return super().__init__(model=model, initial_models=initial_models, *args, **kwargs)

    def _rbf_kernel_matrix(self, X1, X2, length_scale, variance=1.0):
        def sq_dist(x1, x2):
            return jnp.sum((x1 - x2) ** 2)
        def kernel_fn(a, b):
            return variance * jnp.exp(-0.5 * sq_dist(a, b) / (length_scale**2))
        return vmap(lambda x1: vmap(lambda x2: kernel_fn(x1, x2))(X2))(X1)

    def _compute_log_marginal_likelihood(self, length_scale, X, Y, noise_var=1e-5):
        N = X.shape[0]
        K = self._rbf_kernel_matrix(X, X, length_scale)
        K = K + jnp.eye(N) * noise_var  # Add jitter for stability
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), Y)
        data_fit = -0.5 * jnp.sum(Y * alpha)
        log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        complexity = -0.5 * log_det_K * Y.shape[1] # Scale by number of output dims
        return data_fit + complexity

    def _fit_length_scale(self, X, Y):
        candidates = jnp.logspace(jnp.log10(0.01), jnp.log10(2.0), 50)
        evaluate_lml = vmap(lambda l: self._compute_log_marginal_likelihood(l, X, Y))
        lml_scores = evaluate_lml(candidates)
        best_idx = jnp.argmax(jnp.nan_to_num(lml_scores, nan=-jnp.inf))
        return candidates[best_idx]

    def _generate(self, N: int, d: int, samples: jnp.ndarray, features: jnp.ndarray, key=None) -> jnp.ndarray | None:
        # Ensure JAX arrays
        X_train = jnp.asarray(samples)      # (M, d)
        Y_train = jnp.asarray(features)     # (M, ...)

        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]
        else:
            Y_train = Y_train.reshape(Y_train.shape[0], -1)

        # --- STEP 1: PREPROCESS FEATURES ---
        y_mean = jnp.mean(Y_train, axis=0)
        y_std = jnp.std(Y_train, axis=0) + 1e-6
        Y_norm = (Y_train - y_mean) / y_std

        # --- STEP 2: FIT HYPERPARAMETERS (Using Y) ---
        best_length_scale = self._fit_length_scale(X_train, Y_norm)

        # --- STEP 3: PREPARE PREDICTIVE MODEL ---
        K_xx = self._rbf_kernel_matrix(X_train, X_train, best_length_scale)
        K_xx = K_xx + jnp.eye(K_xx.shape[0]) * 1e-5
        L = jnp.linalg.cholesky(K_xx)

        # --- STEP 4: CRUDE GRID SEARCH FOR MAX VARIANCE ---
        n_candidates = 5000 * d 
        key, subkey = random.split(key)
        X_candidates = random.uniform(subkey, shape=(n_candidates, d), minval=0.0, maxval=1.0)
        K_sx = self._rbf_kernel_matrix(X_train, X_candidates, best_length_scale)
        v = jax.scipy.linalg.solve_triangular(L, K_sx, lower=True)
        variance_reduction = jnp.sum(v**2, axis=0)
        pred_variance = 1.0 - variance_reduction
        pred_variance = jnp.maximum(pred_variance, 0.0)

        # --- STEP 5: SELECT NEXT SAMPLES ---
        top_indices = jnp.argsort(pred_variance)[::-1][:N]
        
        # --- STEP 6: CHECK FOR CONVERGENCE ---
        variances = pred_variance[top_indices]
        if jnp.all(variances < self.variance_threshold):
            return None
        
        new_samples = X_candidates[top_indices]
        return new_samples