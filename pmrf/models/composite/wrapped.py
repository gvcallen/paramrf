"""Models that wrap other models to manipulate them."""

from typing import Any, Callable

import equinox as eqx
import parax as prx
from parax.constraints import AbstractConstraint
from distreqx.distributions import AbstractDistribution

from pmrf.models.base import Model
from pmrf.utils import unwrap_self

class Tied(Model):
    """
    A composite model that links or 'ties' fields within a sub-model together.

    This model wraps a base model and allows enforcing a functional relationship
    between a target and a source node within the model. For example,
    this can be used to enforce two parameters to be equal.

    Upon initialization, the target is replaced with a placeholder,
    hiding it from optimizers. When the model is evaluated (via `build`), the
    relationship is unwrapped and the target is resolved.

    Attributes
    ----------
    tied : parax.Tie
        The wrapped parax Tie object handling the parameter linking.

    Examples
    --------
    >>> import pmrf as prf
    >>> from pmrf.models import Resistor, Capacitor
    >>> 
    >>> rc = Resistor(R=50.0) ** Capacitor(C=1.0e-12)
    >>> 
    >>> # Tie the resistor's R to always be 50e12 times the capacitor's C
    >>> tied_rc = Tied(
    ...     model=rc,
    ...     target=lambda m: m.models[0].R,
    ...     source=lambda m: m.models[1].C,
    ...     tie_fn=lambda c: c * 50e12
    ... )
    >>> 
    >>> # The optimizer will now only see the Capacitor's C parameter.
    >>> # When evaluated, R will automatically track C.
    """
    tie: prx.Tie

    def __init__(
        self, 
        model: Model, 
        target: Callable[[Any], Any], 
        source: Callable[[Any], Any], 
        tie_fn: Callable[[Any], Any] = lambda x: x
    ):
        """
        Initialize the Tied model.
        
        Note that if a model is tied that has already been tied,
        the target and source location callables refers to the original, untied model. 

        Parameters
        ----------
        model : Model
            The base RF model whose parameters should be tied.
        target : callable
            A callable (lens) extracting the parameter to be overwritten 
            (e.g., `lambda m: m.resistor.R`).
        source : callable
            A callable (lens) extracting the parameter to draw the value from 
            (e.g., `lambda m: m.capacitor.C`).
        tie_fn : callable, optional
            An optional transformation function applied to the source 
            before injecting it into the target. Defaults to the identity 
            function (`lambda x: x`).
        """
        base_tree = model.tie if isinstance(model, Tied) else model
        
        new_tie = prx.Tie(
            tree=base_tree,
            target=target,
            source=source,
            tie_fn=tie_fn
        )
        
        self.tie = new_tie

    @unwrap_self
    def build(self) -> Model:
        """
        Evaluate and return the underlying tied wrapper structure.

        Returns
        -------
        Model
            The unwrapped `parax.Tie` wrapper containing the resolved relationships.
        """
        return self.tie

    @property
    def model(self) -> Model:
        """
        Returns the underlying model.
        """
        return self.tie.tree
    

class Probabilistic(Model):
    """
    (experimental) A wrapper to make an existing model probabilistic.

    This provides the ability to associate a probability distribution
    with a model or one of its sub-models/parameters after the model
    was created.

    This is a useful for advanced use-cases where you want to attach a distribution
    to an entire model (perhaps overriding previous distributions on lower levels),
    as opposed to more standard cases where you want to model the distributions
    of individual variables (in which case you should likely use `pmrf.Random` instead).

    Attributes
    ----------
    probabilistic : Model | parax.Probabilize
        The updated structure containing the `parax.Probabilize` node.

    Examples
    --------
    >>> from pmrf.models import Probabilistic, Resistor
    >>> from pmrf.distributions import Normal, Joint
    >>> 
    >>> res = Resistor(R=50.0)
    >>> 
    >>> # Use Case 1: Target a specific parameter (leaf)
    >>> prob_res_leaf = Probabilistic(
    ...     model=res, 
    ...     distribution=Normal(loc=50.0, scale=1.0),
    ...     target=lambda m: m.R
    ... )
    >>> 
    >>> # Use Case 2: Wrap the entire model (requires matching distribution tree)
    >>> import equinox as eqx
    >>> dist_tree = Joint(eqx.tree_at(lambda m: m.R, res, Normal(loc=50.0, scale=1.0)))
    >>> prob_res_tree = Probabilistic(
    ...     model=res, 
    ...     distribution=dist_tree,
    ... )
    """
    probabilistic: prx.Probabilize | Model

    def __init__(
        self, 
        model: Model, 
        distribution: AbstractDistribution,
        target: Callable[[Any], Any] = lambda m: m,
        constraint: AbstractConstraint | None = None,
    ):
        """
        Initialize the Probabilistic model.

        Parameters
        ----------
        model : Model
            The base model to wrap.
        distribution : AbstractDistribution or tuple of AbstractDistribution
            The probability distribution(s) to associate with the target(s).
        target : callable, optional
            A callable (lens) extracting the parameter(s) to make probabilistic 
            (e.g., `lambda m: m.R` or `lambda m: (m.R, m.C)`). Defaults to the 
            identity function.
        constraint : AbstractConstraint or tuple of AbstractConstraint, optional
            Optional constraint(s) for the distribution(s).
        """
        target_vals = target(model)

        # Handle the case where multiple targets are returned as a tuple
        if isinstance(target_vals, tuple):
            if not isinstance(distribution, tuple) or len(distribution) != len(target_vals):
                raise ValueError(
                    "If 'target' returns a tuple, 'distribution' must be a tuple "
                    "of the exact same length."
                )
            
            # Normalize constraints to an iterable of the correct length
            if constraint is None:
                constraint_tup = (None,) * len(target_vals)
            elif not isinstance(constraint, tuple) or len(constraint) != len(target_vals):
                raise ValueError(
                    "If 'target' returns a tuple, 'constraint' must be None or a "
                    "tuple of the exact same length."
                )
            else:
                constraint_tup = constraint

            # Generate a Probabilize node for each target item
            prob_nodes = tuple(
                prx.Probabilize(dist, val, constraint=cons)
                for dist, val, cons in zip(distribution, target_vals, constraint_tup)
            )
            self.probabilistic = eqx.tree_at(target, model, prob_nodes)

        # Handle the standard single-target case
        else:
            if isinstance(distribution, tuple):
                raise ValueError(
                    "Provided a tuple of distributions, but 'target' returned a single node."
                )
            if isinstance(constraint, tuple):
                raise ValueError(
                    "Provided a tuple of constraints, but 'target' returned a single node."
                )

            prob_node = prx.Probabilize(distribution, target_vals, constraint=constraint)
            self.probabilistic = eqx.tree_at(target, model, prob_node)

    @property
    def model(self) -> Model:
        """
        Returns the underlying model (or the probabilistic wrapper if applied to the root).
        """
        return self.probabilistic
    
    @unwrap_self
    def build(self) -> Model:
        """
        Evaluate and return the underlying probabilistic model structure.

        Returns
        -------
        Model
            The updated model containing the `parax.Probabilize` node(s).
        """
        return self.probabilistic

    @property
    def model(self) -> Model:
        """
        Returns the underlying model (or the probabilistic wrapper if applied to the root).
        """
        return self.probabilistic
    
    @unwrap_self
    def build(self) -> Model:
        """
        Evaluate and return the underlying probabilistic model structure.

        Returns
        -------
        Model
            The updated model containing the `parax.Probabilize` node.
        """
        return self.probabilistic