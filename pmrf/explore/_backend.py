"""
Base interface for exploration samplers and results.

This implements a lower-level API similar to Optimistix and Inferix.
"""

import abc
from typing import Any, Callable, Generic, TypeVar, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree, Scalar, ArrayLike, Int
import equinox as eqx
import equinox.internal as eqxi

# --- Type Variables & Aliases ---
T = TypeVar("T")
Y = TypeVar("Y")
Out = TypeVar("Out")
Aux = TypeVar("Aux")
SamplerState = TypeVar("State")
Args: TypeAlias = Any

class Batched(Generic[T]):
    pass

Fn: TypeAlias = Callable[[Y, Args], tuple[Out, Aux]]
NoAuxFn: TypeAlias = Callable[[Y, Args], Out]
MaybeAuxFn: TypeAlias = Fn[Y, Out, Aux] | NoAuxFn[Y, Out]
HypercubeFn: TypeAlias = Callable[[Float[Array, "d"], Args], Y]

class RESULTS(eqxi.Enumeration):
    successful = ""
    max_steps_reached = "The maximum number of steps was reached in the adaptive sample. "
    busy = "The sampler is still busy. "
    
class Samples(eqx.Module, Generic[Y, Out, Aux]):
    """
    The results from an adaptive sample.
    
    Contains the fully padded arrays as required by JAX compilation, but exposes 
    ergonomic `y`, `out`, and `aux` properties to automatically strip padding for the user.
    """
    y_padded: Batched[Y]
    out_padded: Batched[Out]
    aux_padded: Batched[Aux]
    is_valid_padded: Bool[Array, "batch"]
    result: RESULTS
    stats: dict[str, PyTree[ArrayLike]]
    state: Any

    @property
    def y(self) -> Batched[Y]:
        """Returns the unpadded, dynamically sized sample inputs."""
        return jax.tree_map(lambda arr: arr[self.is_valid_padded], self.y_padded)

    @property
    def out(self) -> Batched[Out]:
        """Returns the unpadded, dynamically sized sample responses."""
        return jax.tree_map(lambda arr: arr[self.is_valid_padded], self.out_padded)

    @property
    def aux(self) -> Batched[Aux]:
        """Returns the unpadded, dynamically sized auxiliary data."""
        return jax.tree_map(lambda arr: arr[self.is_valid_padded], self.aux_padded)
    

class AbstractAdaptiveSampler(eqx.Module, Generic[Y, Out, Aux, SamplerState]):
    """Abstract base class for all exploration and adaptive samplers."""
    
    rtol: eqx.AbstractVar[float]
    atol: eqx.AbstractVar[float]
    
    # The maximum number of points this algorithm will generate per step.
    batch_size: eqx.AbstractVar[int] 

    @abc.abstractmethod
    def init(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y0: Y,
        y_init: Batched[Y] | None,
        out_init: Batched[Out] | None,
        aux_init: Batched[Aux] | None,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        key: jax.Array,
        tags: frozenset[object],
    ) -> tuple[Batched[Y], Batched[Out], Batched[Aux], Bool[Array, "batch"], SamplerState]:
        """
        Perform all initial computation needed to initialise the sampler.
        
        Returns
        -------
        tuple[Batched[Y], Batched[Out], Batched[Aux], Bool[Array, "batch"], State]
            Returns the exact initial points to be evaluated and logged, along with 
            an `is_valid` boolean mask identifying non-padded entries.
        """
        
    @abc.abstractmethod
    def step(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Batched[Y],
        out: Batched[Out],
        aux: Batched[Aux],
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        key: jax.Array,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Batched[Y], Batched[Out], Batched[Aux], Bool[Array, "batch"], SamplerState]:
        """
        Perform one iteration of the adaptive sampling loop.
        
        Contract:
        This method must return EXACTLY `self.batch_size` items in the leading 
        dimension of the returned batches. The `is_valid` boolean array MUST accurately 
        reflect which elements in this batch are real samples vs. padding.

        Returns
        -------
        tuple[Batched[Y], Batched[Out], Batched[Aux], Bool[Array, "batch"], State]
            The newly evaluated batch of samples alongside the updated sampler state.
        """

    @abc.abstractmethod
    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Batched[Y],
        out: Batched[Out],
        aux: Batched[Aux],
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        tags: frozenset[object] = frozenset(),
    ) -> tuple[Bool[Array, ""], RESULTS]:
        """
        Determine whether the adaptive sampling has converged based on the CURRENT batch and state.

        Returns
        -------
        tuple[Bool[Array, ""], RESULTS]
            A boolean array indicating whether to terminate (True means stop), 
            alongside the termination results enum.
        """        

    @abc.abstractmethod
    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        hypercube_fn: HypercubeFn[Y],
        y: Batched[Y],
        out: Batched[Out],
        aux: Batched[Aux],
        is_valid: Bool[Array, "batch"],
        args: PyTree,
        options: dict[str, Any],
        state: SamplerState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Batched[Y], Batched[Out], Batched[Aux], Bool[Array, "batch"], dict[str, Any]]:
        """
        Final postprocessing step executed after the main sampling loop exits.

        Returns
        -------
        tuple[Batched[Y], Batched[Out], Batched[Aux], Bool[Array, "batch"], dict[str, Any]]
            The final padded histories, validity mask, and a dictionary of solver statistics.
        """
        

# --- Helper functions for buffer management ---
def _create_buffer(template_batch: PyTree, max_capacity: int):
    """Creates an empty buffer matching the tree structure and dtypes."""
    def make_empty(x):
        # NaN padding for floats, zero padding for ints/bools. 
        # (The is_valid mask is the true source of truth regardless)
        if jnp.issubdtype(x.dtype, jnp.inexact):
            return jnp.full((max_capacity, *x.shape[1:]), jnp.nan, dtype=x.dtype)
        else:
            return jnp.zeros((max_capacity, *x.shape[1:]), dtype=x.dtype)
    return jax.tree_map(make_empty, template_batch)

def _insert_into_buffer(buffer: PyTree, new_batch: PyTree, start_idx: Int[Array, ""]):
    """Dynamically updates the buffer along axis 0 at the calculated step index."""
    return jax.tree_map(
        lambda b, update: jax.lax.dynamic_update_slice_in_dim(b, update, start_idx, axis=0),
        buffer, new_batch
    )


@eqx.filter_jit
def sample(
    fn: MaybeAuxFn[Y, Scalar, Aux],
    solver: AbstractAdaptiveSampler,
    y0: Y,
    key: jax.Array,
    y_init: Batched[Y] | None = None,
    out_init: Batched[Out] | None = None,
    aux_init: Batched[Aux] | None = None,
    args: PyTree[Any] = None,
    options: dict[str, Any] | None = None,
    *,
    has_aux: bool = False,
    max_steps: int = 1024,
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
    hypercube_fn: HypercubeFn[Y] | None = None,
) -> Samples[Y, Out, Aux]:

    options = options or {}
    _fn = fn if has_aux else lambda y, a: (fn(y, a), None)
    _hypercube_fn = hypercube_fn if hypercube_fn is not None else lambda x, a: x

    f_struct, aux_struct = jax.eval_shape(lambda y: _fn(y, args), y0)

    # 1. Initialize solver
    initial_y, initial_out, initial_aux, init_valid, init_state = solver.init(
        fn=_fn, hypercube_fn=_hypercube_fn, y0=y0, y_init=y_init, out_init=out_init,
        aux_init=aux_init, args=args, options=options, f_struct=f_struct,
        aux_struct=aux_struct, key=key, tags=tags,
    )

    # 2. Framework allocates the history buffers
    init_batch_size = jax.tree_util.tree_leaves(initial_y)[0].shape[0]
    max_capacity = init_batch_size + (max_steps * solver.batch_size)

    buffer_y = _create_buffer(initial_y, max_capacity)
    buffer_out = _create_buffer(initial_out, max_capacity)
    buffer_aux = _create_buffer(initial_aux, max_capacity)
    buffer_valid = jnp.zeros((max_capacity,), dtype=jnp.bool_)

    # 3. Inject initial data into the very top of the buffers
    buffer_y = _insert_into_buffer(buffer_y, initial_y, 0)
    buffer_out = _insert_into_buffer(buffer_out, initial_out, 0)
    buffer_aux = _insert_into_buffer(buffer_aux, initial_aux, 0)
    buffer_valid = jax.lax.dynamic_update_slice_in_dim(buffer_valid, init_valid, 0, axis=0)
    
    # Check terminate on initial batch
    init_terminated, init_result = solver.terminate(
        fn=_fn, hypercube_fn=_hypercube_fn, y=initial_y, out=initial_out,
        aux=initial_aux, args=args, options=options, state=init_state, tags=tags
    )

    # 4. The Loop Logic
    def cond_fun(val):
        step_idx, b_y, b_out, b_aux, b_valid, state, is_terminated, result, _ = val
        keep_going = jnp.logical_not(is_terminated) & (step_idx < max_steps)
        return keep_going

    def body_fun(val):
        step_idx, b_y, b_out, b_aux, b_valid, state, is_terminated, result, curr_key = val
        step_key, next_key = jax.random.split(curr_key)
        
        # Solver computes a batch of NEW points
        new_y, new_out, new_aux, new_valid, new_state = solver.step(
            fn=_fn, hypercube_fn=_hypercube_fn, y=b_y, out=b_out, 
            aux=b_aux, args=args, options=options, state=state, key=step_key, tags=tags
        )
        
        # Framework injects the block seamlessly at the correct index
        insert_idx = init_batch_size + (step_idx * solver.batch_size)
        next_b_y = _insert_into_buffer(b_y, new_y, insert_idx)
        next_b_out = _insert_into_buffer(b_out, new_out, insert_idx)
        next_b_aux = _insert_into_buffer(b_aux, new_aux, insert_idx)
        next_b_valid = jax.lax.dynamic_update_slice_in_dim(b_valid, new_valid, insert_idx, axis=0)
        
        # Terminate checks ONLY the new batch and the state, avoiding recompilation on dynamic host loops
        new_is_terminated, new_result = solver.terminate(
            fn=_fn, hypercube_fn=_hypercube_fn, y=new_y, out=new_out, 
            aux=new_aux, args=args, options=options, state=new_state, tags=tags
        )
        
        return (step_idx + 1, next_b_y, next_b_out, next_b_aux, next_b_valid, new_state, new_is_terminated, new_result, next_key)

    # 5. Execute Loop
    init_val = (0, buffer_y, buffer_out, buffer_aux, buffer_valid, init_state, init_terminated, init_result, key)
    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    final_step, final_y, final_out, final_aux, final_valid, final_state, _, final_result, _ = final_val

    # 6. Post Processing
    hit_max = (final_step >= max_steps) & (final_result == RESULTS.busy)
    final_result = jnp.where(hit_max, RESULTS.max_steps_reached, final_result)

    if throw:
        final_result = eqxi.error_if(
            final_result,
            final_result == RESULTS.max_steps_reached,
            "Maximum number of steps reached without convergence."
        )

    post_y, post_out, post_aux, post_valid, stats = solver.postprocess(
        fn=_fn, hypercube_fn=_hypercube_fn, y=final_y, out=final_out, 
        aux=final_aux, is_valid=final_valid, args=args, options=options, 
        state=final_state, tags=tags, result=final_result
    )

    stats["num_steps"] = final_step

    return Samples(
        y_padded=post_y,
        out_padded=post_out,
        aux_padded=post_aux,
        is_valid_padded=post_valid,
        result=final_result,
        stats=stats,
        state=final_state
    )


def host_sample(
    fn: MaybeAuxFn[Y, Scalar, Aux],
    solver: AbstractAdaptiveSampler,
    y0: Y,
    key: jax.Array,
    y_init: Batched[Y] | None = None,
    out_init: Batched[Out] | None = None,
    aux_init: Batched[Aux] | None = None,
    args: PyTree[Any] = None,
    options: dict[str, Any] | None = None,
    *,
    has_aux: bool = False,
    max_steps: int = 100_000, # Much higher limit since we aren't pre-allocating
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
    hypercube_fn: HypercubeFn[Y] | None = None,
) -> Samples[Y, Out, Aux]:
    """
    Host-driven equivalent of `sample`. 
    
    Utilizes a Python `while` loop and dynamic list appending rather than JAX 
    `while_loop` and pre-allocation buffers. Ideal for exploration where 
    the upper bound of steps is unknown or massive.
    """
    options = options or {}
    _fn = fn if has_aux else lambda y, a: (fn(y, a), None)
    _hypercube_fn = hypercube_fn if hypercube_fn is not None else lambda x, a: x

    # We aggressively JIT compile the core logic so the python loop overhead is negligible
    jit_init = eqx.filter_jit(solver.init)
    jit_step = eqx.filter_jit(solver.step)
    jit_terminate = eqx.filter_jit(solver.terminate)
    jit_postprocess = eqx.filter_jit(solver.postprocess)

    f_struct, aux_struct = jax.eval_shape(lambda y: _fn(y, args), y0)

    # 1. Initialize solver
    curr_y, curr_out, curr_aux, curr_valid, state = jit_init(
        fn=_fn, hypercube_fn=_hypercube_fn, y0=y0, y_init=y_init, out_init=out_init,
        aux_init=aux_init, args=args, options=options, f_struct=f_struct,
        aux_struct=aux_struct, key=key, tags=tags,
    )

    # Python lists act as our dynamically growing buffers
    history_y = [curr_y]
    history_out = [curr_out]
    history_aux = [curr_aux]
    history_valid = [curr_valid]

    is_terminated, result = jit_terminate(
        fn=_fn, hypercube_fn=_hypercube_fn, y=curr_y, out=curr_out, 
        aux=curr_aux, args=args, options=options, state=state, tags=tags
    )

    step_idx = 0
    curr_key = key

    # 2. Host Execution Loop
    # JAX arrays must be pulled to the CPU/host to evaluate `if` or `while` conditions
    while step_idx < max_steps and not jax.device_get(is_terminated):
        step_key, curr_key = jax.random.split(curr_key)
        
        curr_y, curr_out, curr_aux, curr_valid, state = jit_step(
            fn=_fn, hypercube_fn=_hypercube_fn, y=curr_y, out=curr_out,
            aux=curr_aux, args=args, options=options, state=state,
            key=step_key, tags=tags
        )
        
        history_y.append(curr_y)
        history_out.append(curr_out)
        history_aux.append(curr_aux)
        history_valid.append(curr_valid)

        is_terminated, result = jit_terminate(
            fn=_fn, hypercube_fn=_hypercube_fn, y=curr_y, out=curr_out, 
            aux=curr_aux, args=args, options=options, state=state, tags=tags
        )
        
        step_idx += 1

    # 3. Dynamic Concatenation
    # We combine all iterations into unified arrays exactly once at the end
    def concat_history(hist):
        return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *hist)
        
    final_y = concat_history(history_y)
    final_out = concat_history(history_out)
    final_aux = concat_history(history_aux)
    final_valid = jnp.concatenate(history_valid, axis=0)

    # 4. Post Processing
    hit_max = (step_idx >= max_steps) & (result == RESULTS.busy)
    final_result = jnp.where(hit_max, RESULTS.max_steps_reached, result)
    
    if throw and jax.device_get(final_result) == RESULTS.max_steps_reached:
        raise RuntimeError("Maximum number of steps reached without convergence.")

    post_y, post_out, post_aux, post_valid, stats = jit_postprocess(
        fn=_fn, hypercube_fn=_hypercube_fn, y=final_y, out=final_out,
        aux=final_aux, is_valid=final_valid, args=args, options=options,
        state=state, tags=tags, result=final_result
    )

    stats["num_steps"] = step_idx

    return Samples(
        y_padded=post_y,
        out_padded=post_out,
        aux_padded=post_aux,
        is_valid_padded=post_valid,
        result=final_result,
        stats=stats,
        state=state
    )