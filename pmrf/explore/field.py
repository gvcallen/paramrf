import abc
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, PyTree

from pmrf.explore.base import (
    AbstractAdaptiveSampler, Y, Out, Aux, Fn, HypercubeFn, SamplerState, RESULTS
)
from pmrf.utils.random import lhs_sample
from pmrf.utils.tree import batched_tree_flatten, batched_tree_unflatten

Field = TypeVar('Field')
    
class AbstractFieldSampler(AbstractAdaptiveSampler):
    """Samples new points at the maxima of a learned scalar field."""
    rtol: float
    atol: float
    batch_size: int = 1
    num_grid_per_dim: int = 1024

    @abc.abstractmethod
    def train_field(self, theta: Array, responses: Array, key: Array) -> Field:
        """
        Train a field on the provided flattened input-output responses.

        Parameters
        ----------
        theta : Array
            The input parameters of shape (N, d).
        responses:
            The parameter responses of shape (..., D).
        key : jax.Array
            Random key for stochastic training.

        Returns
        -------
        Field
            The trained field.
        """     

    @abc.abstractmethod
    def evaluate_field(self, field: Field, theta: Array, key: Array) -> Array:
        """
        Evaluate the scalar field at specified input parameters.

        Parameters
        ----------
        field : Field
            The trained field.
        theta:
            The parmaeters to evaluate the field at of shape (N, d).
        key : jax.Array
            Random key for stochastic evaluation.

        Returns
        -------
        Array
            The field evaluations of shape (N, d).
        """     
    
    @abc.abstractmethod
    def init(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Y,
        out: Out,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        key: jax.Array,
        tags: frozenset[object] = frozenset(),
    ) -> SamplerState:
        pass
        
    @abc.abstractmethod
    def step(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Y,
        out: Out,
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        key: jax.Array,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Y, Out, SamplerState, Aux]:
        key, field_key, grid_key, fn_key = jr.split(key, 4)
        
        # Train the field on the previous responses
        y_array, out_array = batched_tree_flatten(y), batched_tree_flatten(out)
        field = self.train_field(y_array, out_array, field_key)
        
        # Generate Candidate Grid (Hypercube)
        d = y_array.shape[1::].size
        K = self.num_grid_per_dim * d
        U_array = lhs_sample(K, d, grid_key)
        
        # Evaluate Field on Grid
        fn_keys = jr.split(fn_key, K)
        field_values = jax.vmap(lambda u, k: self.evaluate_field(field, u, k))(U_array, fn_keys)

        # Greedy Diversity Selection to get new hypercube proposals
        U_array_next, _ = self._select_field_points(self.batch_size, U_array, field_values)
        
        # Unflatten U_array_next and get y_new
        U_next = batched_tree_unflatten(U_array_next, y)
        y_new = hypercube_fn(U_next)
        
        # Evaluate the physical model
        out_new, aux = fn(y_new)
        
        # Stack the responses
        y_return = jnp.vstack((y, y_new))
        out_return = jnp.vstack((out, out_new))
        
        # Return the new y and out. Pass the field and its values as state
        return y_return, out_return, (field, field_values), aux

    @abc.abstractmethod
    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Y,
        out: Out,        
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return False, RESULTS.busy

    @abc.abstractmethod
    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Y,
        out: Out,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Out, Aux, dict[str, Any]]:
        return y, out, aux, {}
    

    def _select_field_points(self, N: int, points: Array, values: Array) -> tuple[Array, Array]:
        """Selects N points iteratively using a penalized greedy strategy."""
        if N >= len(values): return points, values
        if N == 1:
            best_idx = jnp.argmax(values)
            return points[best_idx][None, :], values[best_idx].reshape(1)

        p_min, p_max = jnp.min(points, axis=0), jnp.max(points, axis=0)
        p_range = jnp.where((p_max - p_min) == 0, 1.0, p_max - p_min) 
        norm_points = (points - p_min) / p_range

        v_min, v_max = jnp.min(values), jnp.max(values)
        if v_max == v_min: return points[:N], values[:N]
            
        scores = (values - v_min) / (v_max - v_min)
        L = 0.1 * jnp.sqrt(points.shape[1]) 

        selected_indices = []
        for _ in range(N):
            best_idx = jnp.argmax(scores)
            selected_indices.append(best_idx)

            if len(selected_indices) < N:
                dist_sq = jnp.sum((norm_points - norm_points[best_idx])**2, axis=1)
                penalty = 1.0 - jnp.exp(-dist_sq / (2 * L**2))
                scores = scores * penalty
                scores = scores.at[best_idx].set(-jnp.inf)

        idx_array = jnp.array(selected_indices)
        return points[idx_array], values[idx_array]    