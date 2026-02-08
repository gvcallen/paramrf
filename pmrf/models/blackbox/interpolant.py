from typing import Self, Callable

import jax.numpy as jnp
import equinox as eqx

from pmrf.models.blackbox.blackbox import BlackBox, SupervisedBlackBox
from pmrf.models.model import Model
from pmrf.parameters import Parameter
from pmrf.frequency import Frequency
from pmrf._util import field
from jax.scipy.linalg import cho_factor, cho_solve

class LatentBlackBox(SupervisedBlackBox):
    """A model where the S-parameters are a calculated using an encoder-decoder architecture.
    - The model is a function of P parameters.
    - The encoder is an arbitrary equinox Module with P inputs and D outputs.
    - The decoder is a ParamRF BlackBox model with D parameters.

    """
    # The current parameters of length P
    params: Parameter
    
    # The latent decoder. This must have D flat parameters
    decoder: BlackBox = field(static=True)
    
    def encode(self, params) -> jnp.ndarray:
        pass
    
    def forward(self) -> jnp.ndarray:
        # The forward model, which produces a sample for the current parameters
        return self.decoder.with_params(self.encode(self.flat_param_values())).forward()
    
    # def from_samples(cls, params: jnp.ndarray, samples: jnp.ndarray, decoder: BlackBox, frequency: Frequency, property='s', **kwargs) -> Self:
    #     # Transform the frequency-domain samples back into the latent domain using the decoder's inverse
    #     latent_samples = jnp.array([decoder.inverse(sample) for sample in samples])
    #     return cls(params=params[0], param_samples=params, latent_samples=latent_samples, decoder=decoder, frequency=frequency, feature=property, **kwargs)
              
    
# class KrigingLatentInterpolator(Latent):
#     """A latent interpolator where the interpolation is done using Kriging (Gaussian process regression).
#     """
#     # Hyper-parameters
#     length_scale: float = 1.0
#     noise_variance: float = 1e-6 # Small jitter for numerical stability (regularization)

#     # Pre-calculated weights matrix (alpha) for the GP mean prediction
#     # Shape will be (N, D)
#     weights: jnp.ndarray = field(static=True, init=False)
    
#     def __post_init__(self):
#         """
#         Train the Gaussian Process (Kriging model).
#         Since we are ignoring hyper-parameter tuning, this step simply calculates 
#         the weights alpha = (K + sigma^2 * I)^-1 * Y
#         """
#         # X shape: (N, P)
#         X = self.param_samples
#         # Y shape: (N, D)
#         Y = self.latent_samples

#         # 1. Compute pairwise squared euclidean distances for the kernel matrix
#         # efficient vectorization: (x-y)^2 = x^2 + y^2 - 2xy
#         # shape (N, N)
#         diffs = X[:, None, :] - X[None, :, :]
#         sq_dist_matrix = jnp.sum(diffs**2, axis=-1)

#         # 2. Compute the RBF Kernel Matrix (Covariance Matrix)
#         # K[i,j] = exp(-||x_i - x_j||^2 / (2 * l^2))
#         K = jnp.exp(-sq_dist_matrix / (2 * self.length_scale**2))

#         # 3. Add noise variance to diagonal for stability (K + sigma_n^2 * I)
#         K_y = K + self.noise_variance * jnp.eye(K.shape[0])

#         # 4. Solve for weights (alpha)
#         # We use Cholesky decomposition for numerical stability
#         # Weights shape: (N, D)
#         c, lower = cho_factor(K_y, lower=True)
#         self.weights = cho_solve((c, lower), Y)
    
#     def forward(self) -> jnp.ndarray:
#         # 1. Get current parameters x_star. Shape (P,)
#         x_star = self.flat_param_values()
        
#         # 2. Compute kernel vector k_star between x_star and all training samples X
#         # shape (N,)
#         diff = self.param_samples - x_star # Broadcasting (N, P) - (P,)
#         sq_dist_vector = jnp.sum(diff**2, axis=-1)
#         k_star = jnp.exp(-sq_dist_vector / (2 * self.length_scale**2))

#         # 3. Predict the latent mean
#         # f_star = k_star^T * weights
#         # (N,) dot (N, D) -> (D,)
#         predicted_latent_params = jnp.dot(k_star, self.weights)

#         # 4. Decode the latent parameters to frequency response
#         # We update the decoder with the interpolated latent parameters and run it
#         return self.decoder.with_params(predicted_latent_params).forward()