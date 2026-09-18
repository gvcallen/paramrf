r"""
Non-uniform transmission lines: a uniform line whose parameters vary along it.
"""
from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from pmrf.models.base import Model
from pmrf.models.adapters.delegated import AbstractBuilder
from pmrf.models.components.lines.base import TransmissionLine
from pmrf.models.components.lines.profiles import AbstractProfile
from pmrf.modules.base import Module
from pmrf.models.composite.interconnected.cascade import RepeatedCascade
from pmrf.parameters import (
    Param,
    is_param,
    tree_param_paths,
    update,
)
from pmrf.types import ArrayLike
from pmrf.utils import field, replace

#: Keywords that always belong to the container rather than to the base line. When the
#: base is given as a class, its plain keywords are forwarded to it, and a base class
#: declaring a field of one of these names cannot be told apart from the container's own
#: keyword, so the collision is rejected rather than guessed at. ``extrapolate`` is
#: reserved ahead of the Richardson extrapolation that follows this ticket: the name has
#: to belong to the container from the start, or adding it later would silently take a
#: keyword a base class had been using.
RESERVED_KEYWORDS = ('n', 'extrapolate', 'name', 'metadata')

#: The base parameter that is the line's total length. It is divided by the section
#: count internally and is never profilable.
LENGTH = 'length'

#: The `fnmatch` metacharacters. A target containing one is a glob, and is rejected.
_GLOB_CHARACTERS = '*?['


class ProfiledLine(TransmissionLine, AbstractBuilder):
    r"""
    A line whose parameters vary along it, following profiles.

    A **base** uniform line, plus a mapping from parameter **targets** on it to the
    :class:`~pmrf.models.AbstractProfile` driving each one. The line evaluates as a
    cascade of `n` uniform sections whose profiled parameters are sampled at the
    section midpoints, and every other parameter is shared by every section.

    ``t = 0`` is at port 1 and ``t = 1`` is at port 2, as on
    :class:`~pmrf.models.AbstractProfile`. Reversing a taper swaps the profile's
    coefficients; there is no orientation flag.

    Two ways in, one representation
    -------------------------------
    The base may be a constructed line, or a class plus the keywords to build it::

        ProfiledLine(MicrostripLine, {'w': taper}, h=1.6e-3, length=50e-3)
        ProfiledLine(MicrostripLine(w=4e-3, h=1.6e-3, length=50e-3), {'w': taper})

    The keywords of the class form are forwarded to the base, except for the reserved
    container keywords ``n``, ``extrapolate``, ``name`` and ``metadata``, which always
    belong to the container. A base class declaring a field of one of those names
    raises rather than being guessed at. A base built this way is deliberately unnamed,
    so its parameter names flatten to the container's root.

    Targets
    -------
    A target is an **exact dotted path** naming a :class:`pmrf.Param` on the base, as
    :func:`pmrf.params` gives it: ``'w'``, ``'substrate.dielectric.ep_r'``. Globs,
    sequences and callables are rejected, because a glob matching three parameters
    would silently create three independent profiles that happen to share one shape
    object, which is never what was meant.

    The base tree is left exactly as built: profiles are never substituted into it and
    are never declared as :func:`pmrf.param` fields, both of which would bypass the
    driven field's own converter and constraint (ADR-0004).

    Length
    ------
    `length` is the base's own parameter, and it is the **total** length. It is not
    profilable and is rejected as a target: the container divides it by the section
    count internally, so a per-section length is never seen. `ProfiledLine` has no
    `length` field of its own; :attr:`length` forwards read-only to the base.

    Parameters and names
    --------------------
    A profiled target's value on the base is discarded, and the target stops being a
    parameter of the line: it is **shadowed**, so it has no name, is absent from
    :func:`pmrf.params`, and ``pmrf.update(line, {'w': ...})`` raises rather than
    quietly setting a value that changes nothing. The `Param` itself stays where it is
    in the base tree, so each profile's value is still written back through the driven
    field's own converter and constraint. What is fitted instead are the profile's
    coefficients, which the container names under the target: ``'w.start'``, ``'substrate.dielectric.ep_r.end'``.
    The names are assigned by the container rather than falling out of the mapping's
    dict keys, so a non-identifier path never leaks a bracket form into a parameter
    name, and the container's own field layout never appears in one. Globs then do the
    obvious thing: ``'w.*'`` is one target's coefficients and ``'*.start'`` is every
    profile's start.

    Passing a value explicitly for a profiled target raises, in the class form where
    the container can see what was typed; a field default that is discarded does
    not, because the user did not type it.

    Evaluation
    ----------
    `ProfiledLine` is a :class:`~pmrf.models.TransmissionLine` and an
    :class:`~pmrf.models.AbstractBuilder`: :meth:`build` returns a
    :class:`~pmrf.models.RepeatedCascade`, so ``s``, ``a``, ``y``, ``z`` and ``mna``
    all delegate to one implementation and cannot disagree.

    It is deliberately **not** an ``AbstractUniformLine``. A characteristic impedance
    that is silently the value at one position along the taper invites exactly the
    misuse a resolution guard exists to prevent.

    **Mathematical Formulation**

    With $N$ sections of length $h = L/N$, the midpoints
    $t_k = (k + \tfrac{1}{2})/N$ and $\theta_k$ the base parameters with each target
    replaced by its profile's value at $t_k$,

    $$A(f) = \prod_{k=0}^{N-1} A_{\mathrm{base}}(f;\, \theta_k,\, \ell = h)$$

    Sampling at the midpoints and building exact sections is the exponential midpoint
    rule, an order-2 Magnus integrator, so the error is $O(h^2)$ for a profile that is
    $C^2$ in $t$.

    Parameters
    ----------
    base : Model or type[Model]
        The uniform line the profiles vary, or its class. A class is constructed from
        the plain keywords and is left unnamed.
    profiles : Mapping[str, AbstractProfile]
        Exact dotted parameter paths on the base, mapped to the profile driving each.
    n : int, default=64
        The number of uniform sections. Static.
    **base_kwargs
        Keywords forwarded to `base` when it is a class. Rejected when `base` is
        already constructed.

    Raises
    ------
    TypeError
        If `base` is neither a `Model` nor a `Model` subclass, if a target is not a
        string, if a profile is not an `AbstractProfile`, or if keywords are given
        alongside a constructed base.
    ValueError
        If `profiles` is empty, a target is a glob or does not name a `Param` on the
        base, a target is `length` or carries its own scale, a base class collides
        with a reserved container keyword, a reserved keyword is passed through to the
        base, a value is passed for a profiled target, or `n` is not positive.

    See Also
    --------
    AbstractProfile : The shape a target follows.
    RepeatedCascade : What `build` returns.

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 5.8. Wiley.

    Examples
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import ProfiledLine, MicrostripLine, ExponentialProfile

        taper = ProfiledLine(
            MicrostripLine,
            {'w': ExponentialProfile(
                start=prf.Bounded(1e-3, 9e-3, value=2e-3),
                end=prf.Bounded(1e-3, 9e-3, value=8e-3),
            )},
            h=1.6e-3,
            length=50e-3,
            n=64,
        )

        sorted(prf.params(taper, free_only=True))   # ['w.end', 'w.start']
        taper.at(0.5)                               # the base line at mid-taper
        taper.s(prf.Frequency(1, 10, 101, 'ghz'))
    """
    #: The base line, held exactly as built. Profiled targets on it are fixed.
    base: Model

    #: Exact dotted parameter paths on the base, mapped to the profile driving each.
    profiles: dict[str, AbstractProfile]

    #: The number of uniform sections. Static.
    n: int = field(default=64, static=True, kw_only=True)

    #: Keeps the container's own field layout -- `base`, `profiles` -- out of the names
    #: of the parameters below it, so the base line's parameters flatten to this
    #: container's position in the name space and the profile coefficients are named
    #: under their target. See :data:`pmrf.parameters.NAME_TRANSPARENT_MARKER`.
    _pmrf_name_transparent = True

    def __init__(
        self,
        base: Model | type[Model],
        profiles: Mapping[str, AbstractProfile],
        *,
        n: int = 64,
        name: str | None = None,
        metadata: Any = None,
        **base_kwargs: Any,
    ):
        profiles = _checked_profiles(profiles)

        if isinstance(base, type):
            if not issubclass(base, Model):
                raise TypeError(
                    f"The base of a ProfiledLine must be a pmrf.Model or a Model "
                    f"subclass; got the class {base.__name__}."
                )
            _check_reserved_collision(base)
            _check_reserved_keywords(base_kwargs)
            _check_no_values_for_targets(base_kwargs, profiles)
            # Deliberately unnamed: the base is an implementation detail of this
            # container, so its parameter names flatten to the container's root.
            base = base(**base_kwargs)
        elif isinstance(base, Model):
            if base_kwargs:
                raise TypeError(
                    f"Got keyword(s) {sorted(base_kwargs)} alongside an already "
                    f"constructed base line. Keywords are only forwarded when `base` "
                    f"is a class; update a constructed line with pmrf.update instead."
                )
        else:
            raise TypeError(
                f"The base of a ProfiledLine must be a pmrf.Model or a Model "
                f"subclass; got {type(base).__name__}."
            )

        if int(n) < 1:
            raise ValueError(f"A ProfiledLine needs at least one section; got n={n}.")

        _check_targets(base, profiles)

        # A profiled target's value on the base is discarded: the profile supplies it
        # per section. Fixing it keeps the base tree structurally untouched while
        # taking it out of the free set; `shadowed_param_paths` then takes away its
        # name, so an optimiser is never handed a parameter that moves nothing.
        self.base = update(base, list(profiles), fixed=True)
        self.profiles = {
            target: _named_under(profile, target) for target, profile in profiles.items()
        }
        self.n = int(n)
        self.name = name
        self.metadata = metadata

    def shadowed_param_paths(self) -> set[tuple[Any, ...]]:
        """The paths of the profiled targets, which are no longer parameters.

        A profiled target's value is driven by its profile, so the target's own
        parameter does nothing: a fit or a sweep aimed at it would optimise nothing at
        all, which is the failure this hides it from. The `Param` itself stays exactly
        where it is in the base tree, because :meth:`build` writes each profile's value
        back through the driven field's own converter and constraint, and because the
        base tree is left as built (ADR-0004). What it loses is its *name*: it is
        absent from :func:`pmrf.params`, and ``pmrf.update(line, {'w': ...})`` raises
        rather than quietly doing nothing.

        Returns
        -------
        set[tuple]
            JAX key paths, relative to this container, as
            :data:`pmrf.parameters.SHADOWED_PARAMS_METHOD` describes.
        """
        base = (jax.tree_util.GetAttrKey('base'),)
        on_base = tree_param_paths(self.base)
        # A target that no longer resolves has nothing left to shadow. It cannot
        # happen through the constructor, which rejects a target that names no
        # parameter, but the naming layer is not the place to raise about a base tree
        # someone has since restructured.
        return {
            base + tuple(on_base[target][0])
            for target in self.profiles
            if target in on_base
        }

    @property
    def length(self) -> Param:
        """The **total** length of the line: the base's own `length` parameter.

        Read-only. The per-section length is ``length / n`` and is never exposed:
        the base's `length` carries the name and prior the user wrote, and dividing
        it internally keeps a values dict from an earlier fit applicable.
        """
        return self.base.length

    def at(self, t: ArrayLike) -> Model:
        """The base line with the profiled targets substituted at position `t`.

        This is exactly the substitution :meth:`build` performs at the section
        midpoints, exposed for one position, so ``line.at(0.5).zc_and_gammaL(f)`` is
        the local behaviour at mid-taper. Its `length` is the line's **total** length,
        not a section's.

        Parameters
        ----------
        t : ArrayLike
            Normalised position along the line, in $[0, 1]$, with ``t = 0`` at port 1.
            An array of positions gives the whole profile at once, as a base line
            whose profiled parameters carry `t`'s shape.

        Returns
        -------
        Model
            The base line, with each profiled target set to its profile's value at `t`.
        """
        t = jnp.asarray(t)
        values = {target: profile.evaluate(t) for target, profile in self.profiles.items()}
        return update(self.base, values, space='physical')

    def build(self) -> RepeatedCascade:
        """The cascade of `n` uniform sections sampled at the section midpoints."""
        midpoints = (jnp.arange(self.n) + 0.5) / self.n

        values = {
            target: jnp.broadcast_to(jnp.asarray(profile.evaluate(midpoints)), (self.n,))
            for target, profile in self.profiles.items()
        }

        # The base's `length` is the total; each section carries one N'th of it. The
        # arithmetic is on the parameter's physical value, which is also what the
        # length is under a JAX trace, where parameters are already unwrapped.
        section = update(self.base, {LENGTH: self.length / self.n}, space='physical')
        return RepeatedCascade(section, values)


def _checked_profiles(profiles: Mapping[str, AbstractProfile]) -> dict[str, AbstractProfile]:
    """Validates the mapping's shape: string targets to profiles, and not empty."""
    if not isinstance(profiles, Mapping):
        raise TypeError(
            f"A ProfiledLine's `profiles` must be a mapping from parameter targets to "
            f"profiles; got {type(profiles).__name__}."
        )

    profiles = dict(profiles)
    if not profiles:
        raise ValueError(
            "A ProfiledLine needs at least one profiled target. A line with none is "
            "the base line itself."
        )

    for target, profile in profiles.items():
        if not isinstance(target, str):
            raise TypeError(
                f"A ProfiledLine's targets must be exact dotted parameter paths on the "
                f"base, as strings; got {target!r}."
            )
        if any(character in target for character in _GLOB_CHARACTERS):
            raise ValueError(
                f"Target {target!r} looks like a glob. Targets are exact dotted paths "
                f"only: a glob matching several parameters would silently create "
                f"several independent profiles sharing one shape object. Name each "
                f"target and give each its own profile."
            )
        if not isinstance(profile, AbstractProfile):
            raise TypeError(
                f"The profile for target {target!r} must be a pmrf.models."
                f"AbstractProfile; got {type(profile).__name__}. A sequence or a "
                f"callable is not a profile: a profile is a named, parameter-aware "
                f"shape whose coefficients are fitted."
            )

    return profiles


def _check_reserved_keywords(base_kwargs: Mapping[str, Any]) -> None:
    """Rejects a reserved container keyword rather than forwarding it to the base.

    `n`, `name` and `metadata` are named parameters of the constructor and can never
    reach here; `extrapolate` is reserved ahead of its use, so that it is the
    container's from the start rather than being silently passed to the base line.
    """
    reserved = sorted(set(base_kwargs) & set(RESERVED_KEYWORDS))
    if reserved:
        raise ValueError(
            f"The keyword(s) {reserved} belong to ProfiledLine, not to the base line, "
            f"and are never forwarded to it. The container keywords are "
            f"{list(RESERVED_KEYWORDS)}; 'extrapolate' is reserved for the Richardson "
            f"extrapolation of the section count, which is not implemented yet."
        )


def _check_reserved_collision(base: type[Model]) -> None:
    """Rejects a base class declaring a field the container reserves for itself."""
    fields = {f.name for f in base.__dataclass_fields__.values()}
    # `name` and `metadata` are fields of every module, the container's included, so
    # they are reserved without ever being a collision: only the container's own
    # evaluation keywords can genuinely clash with a base class's field.
    shared_with_every_module = {f.name for f in Module.__dataclass_fields__.values()}
    collisions = sorted(fields & set(RESERVED_KEYWORDS) - shared_with_every_module)
    if collisions:
        raise ValueError(
            f"{base.__name__} declares the field(s) {collisions}, which ProfiledLine "
            f"reserves for itself: the container keywords {list(RESERVED_KEYWORDS)} "
            f"always belong to the container, so a keyword of that name cannot be "
            f"forwarded to the base. Construct the base line yourself and pass it in."
        )


def _check_no_values_for_targets(
    base_kwargs: Mapping[str, Any], profiles: Mapping[str, AbstractProfile]
) -> None:
    """Rejects a value typed for a target that is also profiled."""
    # Only the outermost keyword can be checked, which is what the user typed: a
    # nested target's value lives inside a sub-model that was built elsewhere.
    clashes = sorted(set(base_kwargs) & set(profiles))
    if clashes:
        raise ValueError(
            f"Got both a value and a profile for {clashes}. A profiled target's value "
            f"on the base is discarded, and silently discarded input is not worth the "
            f"afternoon it costs; drop the keyword, or drop the profile."
        )


def _check_targets(base: Model, profiles: Mapping[str, AbstractProfile]) -> None:
    """Resolves every target against the base, rejecting `length` and non-parameters."""
    if LENGTH in profiles:
        raise ValueError(
            f"{LENGTH!r} is not profilable. It is the line's *total* length, which "
            f"ProfiledLine divides by the section count internally. To taper the "
            f"electrical length, profile the parameters that set the propagation "
            f"constant instead."
        )

    known = tree_param_paths(base)
    unknown = sorted(target for target in profiles if target not in known)
    if unknown:
        raise ValueError(
            f"Target(s) {unknown} do not name a parameter on "
            f"{type(base).__name__}. Targets are exact dotted paths, as pmrf.params "
            f"gives them. Available: {sorted(known)}."
        )

    # A profile returns a plain physical value: it is never told which parameter it
    # drives, so it cannot know the units that parameter was declared in. A target
    # carrying its own scale would silently reinterpret that value, so it is rejected
    # rather than guessed at.
    targets = {target: known[target][1] for target in profiles}
    scaled = sorted(
        target for target, node in targets.items()
        if is_param(node) and node.scale is not None
    )
    if scaled:
        raise ValueError(
            f"Target(s) {scaled} declare a scale. A profile returns a plain physical "
            f"value in the units of whatever it drives, so a scaled target would "
            f"reinterpret it; write the profile's coefficients in physical units and "
            f"drop the target's scale."
        )


def _named_under(profile: AbstractProfile, target: str) -> AbstractProfile:
    """Names every coefficient of `profile` under `target`, e.g. ``'w.start'``.

    The container names the coefficients itself rather than letting the mapping's dict
    keys be named structurally, so that a non-identifier target such as ``'a b'`` never
    leaks a bracket form into a parameter name, and the container's own field layout
    never appears in one.
    """
    # Resolved up front: renaming a coefficient changes the name it answers to, but
    # not the path `update` reaches it by.
    coefficients = tree_param_paths(profile)
    for coefficient, (_, node) in coefficients.items():
        if not is_param(node):
            continue
        named = replace(node, name=f'{target}.{coefficient}')
        profile = update(profile, coefficient, fn=lambda _, named=named: named)
    return profile
