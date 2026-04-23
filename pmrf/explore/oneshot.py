"""
One-shot sampling algorithms.
"""
import abc
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

from pmrf.explore.base import (
    AbstractAdaptiveSampler, Y, Out, Aux, Fn, HypercubeFn, SamplerState, RESULTS, Batched
)
from pmrf.utils.random import lhs_sample

class AbstractOneShotSampler(AbstractAdaptiveSampler[Y, Out, Aux, None]):
    """
    Abstract class for one-shot samplers (e.g., LHS, random uniform, Sobol).
    
    Generates all points during the first step. Contains no state.
    """
    
    num_samples: int
    
    @abc.abstractmethod
    def generate_hypercube(self, N: int, d: int, key: jax.Array) -> Float[Array, "N d"]:
        """
        Generate N flat points of dimension d within the unit hypercube.

        Parameters
        ----------
        N : int
            The number of points to generate.
        d : int
            The dimensionality of the hypercube.
        key : jax.Array
            Random key for stochastic generation sampling.

        Returns
        -------
        Array
            A 2D array of shape (N, d) bounded between [0, 1].
        """
        
    @property
    def rtol(self):
        return 0.0

    @property
    def atol(self):
        return 0.0
    
    @abc.abstractmethod
    def init(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y0: Y,
        y_init: Batched[Y] | None,
        out_init: Batched[Out] | None,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        key: jax.Array,
        tags: frozenset[object],
    ) -> tuple[Batched[Y], Batched[Out], Aux, SamplerState]:
        # Dynamically infer dimensionality 'd' by flattening y
        leaves, _ = jax.tree_util.tree_flatten(y0)
        d = sum(jnp.size(leaf) for leaf in leaves)

        # 1. Generate points in unit hypercube
        u_points = self.generate_hypercube(self.num_samples, d, key)
        
        # 2. Vectorize mappings to handle the batch dimension
        v_hypercube_fn = jax.vmap(lambda u: hypercube_fn(u, args))
        y_new = v_hypercube_fn(u_points)
        
        # 3. Vectorize response evaluations
        v_fn = jax.vmap(lambda y: fn(y, args))
        out_new, aux = v_fn(y_new)
        
        return y_new, out_new, aux, aux

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Batched[Y],
        out: Batched[Out],
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        key: jax.Array,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Batched[Y], Batched[Out], SamplerState, Aux]:
        aux = state
        return y, out, state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Batched[Y],
        out: Batched[Out],        
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return True, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Batched[Y],
        out: Batched[Out],
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: tuple[Y, Out, Aux],
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Batched[Y], Batched[Out], Aux, dict[str, Any]]:
        """Extracts the final payload directly from the tuple state."""
        return Y, out, aux, {}
    
    
class UniformSampler(AbstractOneShotSampler):
    """Sampler using uniform random sampling."""
    def generate_hypercube(self, N: int, d: int, key: jax.Array) -> jnp.ndarray:
        return jax.random.uniform(key, shape=(N, d))

   
class LatinHypercubeSampler(AbstractOneShotSampler):
    """Sampler using Latin Hypercube Sampling (LHS)."""
    def generate_hypercube(self, N: int, d: int, key: jax.Array) -> jnp.ndarray:
        return lhs_sample(N, d, key)
