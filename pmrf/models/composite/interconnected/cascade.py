"""
Composite models that physically connect ports of other models in series.
"""
import jax
import jax.numpy as jnp
from dataclasses import InitVar
from typing import Literal

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.types import ArrayLike
from pmrf.rf import a2s, s2a, a2mna, s2mna, MNAStamp
from pmrf.rf import cascade_scattering, cascade_abcd
from pmrf.parameters import tree_param_names_to_path, values as _values, update


HUB_Z0 = 50.0 + 0.0j

EVAL_Z0 = 50.0

class Cascade(Model):
    """
    Represents a cascade, or series connection, of two or more `Model` objects.

    This container connects multiple models end-to-end. The output port of
    one model is connected to the input port of the next.

    All models must have 2N-many ports. Ports N to 2*N-1 of the first model
    are connected to ports 0 to N-1 of the second, and so on.

    Any nested `Cascade` instances are automatically flattened to maintain
    a simple, linear chain of models.

    Parameters
    ----------
    cascade : tuple[Model, ...]
        The sequence of models in the cascade.
    method : {'s', 'a'}, default='s'
        The underlying mathematical domain to use for the cascade reduction.
    flatten: bool, default=True
        (experimental) Flattens the cascade into one large cascade if they contain sub-cascades.

    Notes
    -----
    The scattering reduction pins junction modes that are singular and
    unobservable from the external ports, such as the common mode between
    floating networks, and inverts everything else exactly, so S and its JAX
    gradients match `Circuit.from_chain`. A resonance is observable and never
    pinned. A lossless model evaluated exactly at a true resonance frequency
    is still singular; that is physical. See :func:`pmrf.rf.cascade_two_s`.

    Examples
    --------
    Cascading models is most easily done using the `**` operator, which is
    an alias for creating a `Cascade` model.

    >>> import pmrf as prf
    >>> from pmrf.models import Resistor, Capacitor, Inductor

    # Create individual component models
    >>> res = Resistor(50)
    >>> cap = Capacitor(1e-12)
    >>> ind = Inductor(1e-9)

    # Cascade them together in a series R-L-C configuration
    # This is equivalent to Cascade(models=(res, ind, cap))
    >>> rlc_series = res ** ind ** cap

    # Define a frequency axis
    >>> freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')

    # Calculate the S-parameters of the cascaded network
    >>> s_params = rlc_series.s(freq)
    """
    #: The models.
    cascade: tuple[Model, ...]
    
    #: (experimental) Flatten the connections if they contain any sub-circuits
    flatten: bool = field(default=True, static=True, kw_only=True)
    
    #: The cascade reduction algorithm method.
    method: Literal['s', 'a'] = field(default='s', kw_only=True, static=True)

    def __post_init__(self):
        for model in self.cascade:
            if model.nports % 2 != 0:
                raise ValueError('All networks must be 2N-ports for Cascade')
            
    def expand(self):
        from pmrf.models import Circuit
        
        built = Circuit.from_chain(self.cascade)
        port_map = [(built, p) for p in range(self.nports)]
        return port_map, []
            
    @property
    def number_of_ports(self):
        return self.cascade[0].number_of_ports
    
    def flattened(self) -> 'Cascade':
        merged = []
        for model in self.cascade:
            # Only extend if the user has not given it a name or metadata
            if isinstance(model, Cascade) and model.name is None and model.metadata is None:
                merged.extend(model.cascade)
            else:
                merged.append(model)

        return Cascade(merged, method=self.method, flatten=False)

    # --- DATA EVALUATION ---

    def _evaluate_scattering(self, freq: Frequency, z0: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates S-parameters and stacks them into shape (Nf, N_models, P, P)."""
        S_blocks = jnp.stack([c.s(freq, z0=z0) for c in self.cascade], axis=1)
        
        Nf = S_blocks.shape[0]
        n_components = len(self.cascade)
        c_ports = self.cascade[0].nports
        
        # Broadcast port impedances to match shape
        z0_blocks = jnp.broadcast_to(jnp.asarray(z0, dtype=S_blocks.dtype), (Nf, n_components, c_ports))
        return S_blocks, z0_blocks

    def _evaluate_abcd(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates ABCD-parameters and stacks them into shape (Nf, N_models, P, P)."""
        return jnp.stack([c.a(freq) for c in self.cascade], axis=1)

    # --- SIMULATION & CONVERSION ---

    def _solve(self, freq: Frequency, z0: ArrayLike = EVAL_Z0) -> tuple[jnp.ndarray, jnp.ndarray, str]:
        """Dispatches data prep and solving across the active vmapped mathematical method."""
        if self.flatten:
            flat = self
        else:
            flat = self.flattened()

        if flat.method == 's':
            s_blocks, z0_blocks = flat._evaluate_scattering(freq, z0)
            run_vmap = jax.vmap(cascade_scattering, in_axes=(0, 0))
            s_cas, z0_cas = run_vmap(s_blocks, z0_blocks)
            return s_cas, z0_cas, 's'
            
        elif flat.method == 'a':
            a_blocks = flat._evaluate_abcd(freq)
            run_vmap = jax.vmap(cascade_abcd, in_axes=(0,))
            a_cas = run_vmap(a_blocks)
            return a_cas, None, 'a'
            
        else:
            raise ValueError(f"Unknown cascade method: {self.method}")

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        mat, mat_z0, domain = self._solve(freq, z0)
        if domain == 's':
            return mat
        elif domain == 'a':
            return a2s(mat, z0=z0)

    def a(self, freq: Frequency) -> jnp.ndarray:
        mat, mat_z0, domain = self._solve(freq)
        if domain == 's':
            return s2a(mat, z0=mat_z0)
        elif domain == 'a':
            return mat

    def mna(self, freq: Frequency) -> MNAStamp:
        """
        Overrides the generic `Model.mna()` fallback, which prefers `.a()`
        over `.s()` whenever both are overridden.
        """
        if self.nports <= 2:
            return a2mna(self.a(freq))
        return s2mna(self.s(freq, z0=HUB_Z0), z0=HUB_Z0)


class RepeatedCascade(Model):
    r"""
    A cascade of many copies of one model, built under a single vmap.

    One **member** model, plus a mapping from parameter names on it to arrays
    whose leading axis is the **repeat axis**. Parameters the mapping does not
    name are shared by every repeat. The member is traced once, under a vmap over
    that axis, and the sections are reduced with the :mod:`pmrf.rf` routines
    :class:`Cascade` also uses. Writing the same cascade as
    ``Cascade([MicrostripLine(w=w_i, ...) for w_i in widths])`` instead traces the
    member once per section and names it once per section.

    Repeat axis versus batch axis
    -----------------------------
    The **repeat axis** is physical: it indexes sections along the cascade, from
    the section at port 1 to the section at port :math:`N`. It leads each array in
    `values`, is reduced away, and never appears in the result's shape.

    The **batch axis** is the parameter batch dimension used everywhere else in
    ParamRF (:func:`pmrf.sweep`): one model at many parameter settings. It is added
    *outside* this model and survives into the result, so a sweep over a
    `RepeatedCascade` is a sweep over whole cascades, each with its own repeat
    axis. This is why the class is not called ``BatchedCascade``. See ADR-0004.

    Parameter names
    ---------------
    The member keeps its own names under ``model`` (``model.w``), one set for the
    whole cascade. A repeated parameter's value on the member is discarded, so it
    is fixed at construction and leaves the free set rather than offering an
    optimiser a parameter that moves nothing. The repeated values are themselves a
    free, unconstrained parameter, ``values.<name>``. Fitting a *shape* rather than
    the values is what the profiled lines of ADR-0004 are for.

    **Mathematical Formulation**

    With $M$ repeats and $\theta_k$ the parameters of repeat $k$ — the shared
    values, with the named entries replaced by ``values[name][k]``:

    $$A(f) = \prod_{k=0}^{M-1} A_{\mathrm{member}}(f; \theta_k)$$

    or the corresponding ordered star product when `method` is ``'s'``. Repeat $0$
    is at port 1.

    Parameters
    ----------
    model : Model
        The member model, repeated along the cascade. Must be a 2N-port.
    values : dict[str, ArrayLike]
        Parameter names on `model`, as :func:`pmrf.params` gives them, mapped to
        arrays whose leading axis is the repeat axis. All must agree on the length
        of that axis, which is the static repeat count. Values are in declared
        space and are checked against each parameter's constraint.
    method : {'a', 's'}, default='a'
        The domain the reduction runs in. ABCD needs no reference-impedance
        bookkeeping between sections; scattering is for members whose ABCD form is
        ill-conditioned. Scattering junctions are handled as in :class:`Cascade`.

    Raises
    ------
    ValueError
        If `values` is empty, names a parameter the member does not have, or its
        arrays disagree on the length of their leading axis; or if the member is
        not a 2N-port.

    See Also
    --------
    Cascade : The same reduction over a sequence of distinct models.

    Examples
    --------
    .. code-block:: python

        n = 10
        line = MicrostripLine(w=4e-3, h=1.6e-3, length=10e-3 / n)
        taper = RepeatedCascade(line, {'w': jnp.linspace(3e-3, 6e-3, n)})
        sorted(prf.params(taper, free_only=True))   # ['values.w']
    """
    #: The member model, repeated along the cascade.
    model: Model

    #: Parameter names on the member mapped to per-repeat values, repeat axis first.
    values: dict[str, ArrayLike] = field(
        converter=lambda v: {k: jnp.asarray(x) for k, x in dict(v).items()}
    )

    #: The cascade reduction algorithm method.
    method: Literal['a', 's'] = field(default='a', kw_only=True, static=True)

    def __post_init__(self):
        if self.model.nports % 2 != 0:
            raise ValueError('The member of a RepeatedCascade must be a 2N-port.')

        if not self.values:
            raise ValueError(
                'RepeatedCascade needs at least one repeated parameter: the repeat '
                'count is the leading axis length of the values it is given.'
            )

        known = tree_param_names_to_path(self.model)
        unknown = [name for name in self.values if name not in known]
        if unknown:
            raise ValueError(
                f"Unknown parameter name(s) on the member model: {sorted(unknown)}. "
                f"Available names: {sorted(known)}."
            )

        scalar = sorted(name for name, x in self.values.items() if jnp.ndim(x) == 0)
        if scalar:
            raise ValueError(
                f"Repeated values must have a leading repeat axis, but {scalar} "
                f"is scalar. A parameter shared by every repeat belongs on the member "
                f"model, not in the mapping."
            )

        lengths = {name: x.shape[0] for name, x in self.values.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"The repeated values disagree on the length of their repeat axis: "
                f"{lengths}. Every array's leading axis is the repeat axis, so they "
                f"must all have the same length."
            )

        # A repeated parameter's value on the member is discarded: the mapping
        # supplies it per repeat. Leaving it free would hand an optimiser a
        # parameter that moves nothing, so it is fixed here and stays out of the
        # free parameter set.
        self.model = update(self.model, list(self.values), fixed=True)

    @property
    def repeats(self) -> int:
        """The number of repeats: the length of the repeat axis. Static."""
        return next(iter(self.values.values())).shape[0]

    @property
    def number_of_ports(self):
        return self.model.number_of_ports

    # --- DATA EVALUATION ---

    def _sections(self, evaluate) -> jnp.ndarray:
        """
        Applies `evaluate` to every repeat under one vmap, stacking along axis 0.

        The values are written in once, unbatched, so each constraint is checked
        against the whole repeat axis there rather than re-traced per repeat. The
        vmap then writes the raw values that check produced, needing no further
        validation.
        """
        validated = update(self.model, self.values)
        raw = _values(validated, list(self.values), space='raw')

        def section(raw_slice):
            return evaluate(update(self.model, raw_slice, space='raw'))

        return jax.vmap(section)(raw)

    def _evaluate_scattering(self, freq: Frequency, z0: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates S-parameters and stacks them into shape (Nf, M, P, P)."""
        s_sections = self._sections(lambda m: m.s(freq, z0=z0))
        s_blocks = jnp.moveaxis(s_sections, 0, 1)

        Nf, n_sections, c_ports = s_blocks.shape[:3]
        z0_blocks = jnp.broadcast_to(
            jnp.asarray(z0, dtype=s_blocks.dtype), (Nf, n_sections, c_ports)
        )
        return s_blocks, z0_blocks

    def _evaluate_abcd(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates ABCD-parameters and stacks them into shape (Nf, M, P, P)."""
        return jnp.moveaxis(self._sections(lambda m: m.a(freq)), 0, 1)

    # --- SIMULATION & CONVERSION ---

    def _solve(self, freq: Frequency, z0: ArrayLike = EVAL_Z0) -> tuple[jnp.ndarray, jnp.ndarray, str]:
        """Dispatches data prep and solving across the active vmapped mathematical method."""
        if self.method == 's':
            s_blocks, z0_blocks = self._evaluate_scattering(freq, z0)
            s_cas, z0_cas = jax.vmap(cascade_scattering, in_axes=(0, 0))(s_blocks, z0_blocks)
            return s_cas, z0_cas, 's'

        elif self.method == 'a':
            a_blocks = self._evaluate_abcd(freq)
            a_cas = jax.vmap(cascade_abcd, in_axes=(0,))(a_blocks)
            return a_cas, None, 'a'

        else:
            raise ValueError(f"Unknown cascade method: {self.method}")

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        mat, mat_z0, domain = self._solve(freq, z0)
        if domain == 's':
            return mat
        elif domain == 'a':
            return a2s(mat, z0=z0)

    def a(self, freq: Frequency) -> jnp.ndarray:
        mat, mat_z0, domain = self._solve(freq)
        if domain == 's':
            return s2a(mat, z0=mat_z0)
        elif domain == 'a':
            return mat

    def mna(self, freq: Frequency) -> MNAStamp:
        """
        Overrides the generic `Model.mna()` fallback, which prefers `.a()`
        over `.s()` whenever both are overridden.
        """
        if self.nports <= 2:
            return a2mna(self.a(freq))
        return s2mna(self.s(freq, z0=HUB_Z0), z0=HUB_Z0)
