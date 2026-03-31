"""
An abstract object that represents an arbitrary evaluation of a model over frequency.
"""

from __future__ import annotations
import operator
from typing import Any
import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency

class Evaluator(prx.Module):
    """
    A callable for paremetric, composable, frequency-dependent model feature extraction.
    
    Supports operator overloading to compose evaluators into complex graphs.
    """
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError

    # --- Arithmetic Operators ---

    def __add__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.add)

    def __sub__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.sub)

    def __mul__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.mul)

    def __truediv__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.truediv)

    def __pow__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.pow)

    # --- Reverse Arithmetic (for <scalar> + <Evaluator>) ---

    def __radd__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=other, right=self, fn=operator.add)

    def __rsub__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=other, right=self, fn=operator.sub)

    def __rmul__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=other, right=self, fn=operator.mul)

    # --- Comparison Operators (Useful for custom logic) ---

    def __gt__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.gt)

    def __lt__(self, other: Any) -> Evaluator:
        from pmrf.evaluators import Binary
        return Binary(left=self, right=other, fn=operator.lt)