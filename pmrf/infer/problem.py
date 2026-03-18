import dataclasses
from typing import Callable, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import flatten_util

from pmrf.core import Model
from pmrf.core import Frequency
from pmrf.parameters import is_valid_param
from pmrf.distributions.joint import JointDistribution
from pmrf.likelihood import Likelihood

class InferenceProblem:
    """
    Translates a PyTree Model and a log-likelihood function into a flat Bayesian sampling problem.
    """
    x0: jnp.ndarray
    param_names: list[str]
    
    icdf_fn: Callable[[jnp.ndarray], jnp.ndarray]
    log_likelihood_fn: Callable[[Model, Frequency], jnp.ndarray]
    flat_log_likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray]
    reconstruct_fn: Callable[[jnp.ndarray], tuple[Model, Callable]]

    def __init__(
        self,
        model: Model,
        log_likelihood_fn: Callable[[Model, Frequency], jnp.ndarray] | Sequence,
        frequency: Frequency,
    ):
        # 1. Resolve Likelihood Sequence
        if isinstance(log_likelihood_fn, Sequence):
            flat_log_likelihood_fn = CombinedLikelihood(tuple(log_likelihood_fn))
        else:
            flat_log_likelihood_fn = log_likelihood_fn

        # 2. Extract Model Parameters
        params_tree, static_tree = model.partition()
        model_flat_x, model_unravel_fn = flatten_util.ravel_pytree(params_tree)
        model_names = model.flat_param_names()
        model_dist = model.distribution()

        # 3. Extract Likelihood Parameters (Dynamic PyTree Search)
        ll_leaves, ll_treedef = jax.tree_util.tree_flatten(flat_log_likelihood_fn, is_leaf=is_valid_param)
        
        ll_flat_x_list, ll_names, ll_dists = [], [], []
        for i, leaf in enumerate(ll_leaves):
            if is_valid_param(leaf) and not leaf.fixed:
                val_flat, _ = flatten_util.ravel_pytree(leaf.value)
                ll_flat_x_list.append(val_flat)
                ll_names.append(leaf.name if leaf.name else f"likelihood_param_{i}")
                ll_dists.append(leaf.distribution)

        ll_flat_x = jnp.concatenate(ll_flat_x_list) if ll_flat_x_list else jnp.array([])

        # 4. Define Unified Reconstructor
        def reconstruct_fn(flat_x: jnp.ndarray) -> tuple[Model, Callable]:
            model_x = flat_x[:len(model_flat_x)]
            ll_x = flat_x[len(model_flat_x):]

            rec_model = eqx.combine(model_unravel_fn(model_x), static_tree)

            if len(ll_x) > 0:
                new_ll_leaves = []
                idx = 0
                for leaf in ll_leaves:
                    if is_valid_param(leaf) and not leaf.fixed:
                        size = jnp.size(leaf.value)
                        new_val = ll_x[idx:idx+size].reshape(jnp.shape(leaf.value))
                        new_ll_leaves.append(dataclasses.replace(leaf, value=new_val))
                        idx += size
                    else:
                        new_ll_leaves.append(leaf)
                rec_ll = jax.tree_util.tree_unflatten(ll_treedef, new_ll_leaves)
            else:
                rec_ll = flat_log_likelihood_fn
                
            return rec_model, rec_ll

        # 5. Define Flat Log-Likelihood
        @eqx.filter_jit
        def flat_log_likelihood_fn(flat_x: jnp.ndarray) -> jnp.ndarray:
            rec_model, rec_ll = reconstruct_fn(flat_x)
            return rec_ll(rec_model, frequency)

        # 6. Define Unified Prior (ICDF)
        if ll_dists:
            ll_joint_dist = JointDistribution(
                distributions=ll_dists,
                distribution_names=[[n] for n in ll_names],
                param_names=ll_names
            )
            
        @eqx.filter_jit
        def icdf_fn(u: jnp.ndarray) -> jnp.ndarray:
            theta_model = model_dist.icdf(u[:len(model_names)])
            if ll_dists:
                theta_ll = ll_joint_dist.icdf(u[len(model_names):])
                return jnp.concatenate([theta_model, theta_ll])
            return theta_model

        # 7. Final Assignment
        self.icdf_fn = icdf_fn
        self.flat_log_likelihood_fn = flat_log_likelihood_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.reconstruct_fn = reconstruct_fn
        self.param_names = model_names + ll_names
        self.x0 = jnp.concatenate([model_flat_x, ll_flat_x])