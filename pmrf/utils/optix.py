from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic

import equinox as eqx
import jax

__all__ = ["Lens", "FreeLens", "focus"]

# Re-implements jax-optics but in Python 3.11 using Typevars for compatibility
T = TypeVar("T")
S = TypeVar("S")
I = TypeVar("I")


@dataclass(frozen=True)
class _LensIndexingHelper(Generic[T, S]):
    lens: Lens[T, S]

    def __getitem__(self, index) -> Lens[T, S]:
        return _ArrayLens(self.lens, index)


@dataclass(frozen=True)
class _FreeLensIndexingHelper(Generic[T, S]):
    lens: FreeLens[T, S]

    def __getitem__(self, index) -> FreeLens[T, S]:
        return _FreeArrayLens(self.lens, index)


class Lens(Protocol[T, S]):
    """A protocol for lenses."""

    def get(self) -> S:
        """Get the value of the focus in the object."""
        ...

    def set(self, val: S) -> T:
        """Set the value of the focus in the object.

        Args:
            val: The new value to set.

        Returns:
            A modified copy of the object.
        """
        ...

    def apply(self, update: Callable[[S], S]) -> T:
        """Apply a function to the focused value in the object.

        Args:
            update: The function to apply to the focused value.

        Returns:
            A modified copy of the object
        """
        ...

    @property
    def at(self) -> _LensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        ...


class FreeLens(Protocol[T, S]):
    """A protocol for free lenses."""

    def bind(self, obj: T) -> Lens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        ...

    @property
    def at(self) -> _FreeLensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            An unbound lens.
        """
        ...


@dataclass(frozen=True)
class _FreeLens(Generic[T, S]):
    """A lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.
    """

    where: Callable[[T], S]

    def bind(self, obj: T) -> Lens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return _BoundLens(obj, self.where)

    @property
    def at(self) -> _FreeLensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            An unbound lens.
        """
        return _FreeLensIndexingHelper(self)


@dataclass(frozen=True)
class _BoundLens(Generic[T, S]):
    """A lens that focuses on a value in a bound object.

    Args:
        obj: The object to focus on.
        where: A function that retrieves the focused value from the object.
    """

    obj: T
    where: Callable[[T], S]

    def get(self) -> S:
        """Get the value of the focus in the object.

        Returns:
            The focused value.
        """
        return self.where(self.obj)

    def set(self, val: S) -> T:
        """Set the value of the focus in the object.

        Args:
            val: The new value to set.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, self.obj, replace=val)

    def apply(self, update: Callable[[S], S]) -> T:
        """Apply a function to the focused value in the object.

        Args:
            update: The function to apply to the focused value.

        Returns:
            A modified copy of the object.
        """
        return eqx.tree_at(self.where, self.obj, replace=update(self.get()))

    @property
    def at(self) -> _LensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        return _LensIndexingHelper(self)


@dataclass(frozen=True)
class _ArrayLens(Generic[T, S, I]):
    """A lens that focuses on an index in an array in a bound object.

    Args:
        lens: The lens to focus on.
        index: The index to focus on.
    """

    lens: Lens[T, S]
    index: I

    def get(self, **kwargs) -> S:
        """Get the value of the focus in the object.

        Args:
            **kwargs: Additional arguments to pass to the jax getter.

        Returns:
            The focused value.
        """

        def _getter(x: jax.Array) -> jax.Array:
            return x.at[self.index].get(**kwargs)

        return jax.tree.map(_getter, self.lens.get())

    def set(self, val: S, **kwargs) -> T:
        def _setter(x: jax.Array, y: jax.Array) -> jax.Array:
            return x.at[self.index].set(y, **kwargs)

        return self.lens.set(jax.tree.map(_setter, self.lens.get(), val))

    def apply(self, update: Callable[[S], S], **kwargs) -> T:
        return self.set(update(self.get(**kwargs)), **kwargs)

    @property
    def at(self) -> _LensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            A bound lens.
        """
        raise ValueError("Cannot index and already indexed lens.")


@dataclass(frozen=True)
class _FreeArrayLens(Generic[T, S, I]):
    """A lens that focuses on an index in an array in a bound object.

    Args:
        lens: The lens to focus on.
        index: The index to focus on.
    """

    lens: _FreeLens[T, S]
    index: I

    def bind(self, obj: T) -> Lens[T, S]:
        """Bind the lens to an object.

        Args:
            obj: The object to bind to.

        Returns:
            A bound lens.
        """
        return _ArrayLens(self.lens.bind(obj), self.index)

    @property
    def at(self) -> _FreeLensIndexingHelper[T, S]:
        """Focus on an index in an array in the object.

        Args:
            index: The index to focus on.

        Returns:
            An unbound lens.
        """
        raise ValueError("Cannot index and already indexed lens.")


@dataclass(frozen=True)
class _Focused(Generic[T]):
    """An object that can be focused on.

    Args:
        obj: The object to focus on.
    """

    obj: T

    def at(self, where: Callable[[T], S]) -> Lens[T, S]:
        """Focus on a value in the object.

        Args:
            where: A function that retrieves the focused value from the object.

        Returns:
            A bound lens.
        """
        return _BoundLens(self.obj, where)


def focus(obj: T) -> _Focused[T]:
    """Focus on an object."""
    return _Focused(obj)


def lens(where: Callable[[T], S]) -> _FreeLens[T, S]:
    """Create a lens that focuses on a value in an object.

    Args:
        where: A function that retrieves the focused value from the object.

    Returns:
        An unbound lens.
    """
    return _FreeLens(where)