"""Models that wrap other models to manipulate them."""

from typing import Any, Callable
import parax
from pmrf.models.base import Model


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
    tied_wrapper : parax.Tie
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

    wrapped: parax.Tie

    def __init__(
        self, 
        model: Model, 
        target: Callable[[Any], Any], 
        source: Callable[[Any], Any], 
        tie_fn: Callable[[Any], Any] = lambda x: x
    ):
        """
        Initialize the Tied model.

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
        base_tree = model.wrapped if isinstance(model, Tied) else model
        self.wrapped = parax.Tie(
            tree=base_tree,
            target=target,
            source=source,
            tie_fn=tie_fn
        )

    def build(self) -> Model:
        """
        Evaluate and return the underlying tied wrapper structure.

        Returns
        -------
        Model
            The unwrapped `parax.Tie` wrapper containing the resolved relationships.
        """
        return self.wrapped

    @property
    def model(self) -> Model:
        """
        Returns the underlying model that has tied placeholders.
        """
        return self.wrapped.tree