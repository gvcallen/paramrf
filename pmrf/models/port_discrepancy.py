"""Port discrepancies carried by a component into a circuit."""

from abc import abstractmethod
import re

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.models.adapters.derived import Derived, _unnamed
from pmrf.modules.base import Module
from pmrf.parameters import Param, param
from pmrf.rf import renormalize_s
from pmrf.types import ArrayLike
from pmrf.utils import field, freeze, unwrap, unwrap_self


class AbstractPortDiscrepancy(Module):
    """A port discrepancy returning a complex ``(nf, n, n)`` matrix."""

    @abstractmethod
    def __call__(self, frequency: Frequency) -> Array:
        """Evaluate the discrepancy at the requested frequencies."""
        raise NotImplementedError


def _block_indices(block: str) -> tuple[str, int, int]:
    if not isinstance(block, str) or re.fullmatch(r'(?:[1-9][1-9]|[sa][1-9][1-9])', block) is None:
        raise ValueError(f"Invalid port discrepancy block {block!r}; use 'ii', 'sij', or 'aij'.")
    i, j = int(block[-2]) - 1, int(block[-1]) - 1
    kind = block[0] if len(block) == 3 else 'reflection'
    if (kind == 'reflection' and i != j) or (kind != 'reflection' and i <= j):
        raise ValueError(f"Invalid port discrepancy block {block!r}; transmission requires i > j.")
    return kind, i, j


class GridPortDiscrepancy(AbstractPortDiscrepancy):
    r"""Dense port discrepancy on one frozen frequency grid.

    ``values`` is real with shape ``(n_blocks, 2, nf)``: block, real/imaginary
    part, frequency. A reference fit's event transform must produce blocks in
    the order named by ``blocks``, so its joint prediction maps onto these
    values unchanged. Port indices in block names are one-based single digits.

    Evaluation requires exactly the stored grid, eagerly and under JIT. There
    is no interpolation: make a joint prediction on a new grid instead, since
    interpolating a GP prediction underestimates variance between its points.

    **Mathematical Formulation**

    $$\delta_{ii}=v_{ii},\qquad
      \delta_{ij}=v^s_{ij}+v^a_{ij},\qquad
      \delta_{ji}=v^s_{ij}-v^a_{ij}\quad (i>j).$$

    An absent block contributes zero. Omit antisymmetric blocks for reciprocal
    components.
    """

    #: Real block × real/imaginary part × frequency values.
    values: Param = param()
    #: Frozen grid at which the joint prediction was made.
    frequency: Frequency = field(converter=freeze)
    #: Ordered reflection, symmetric and antisymmetric block names.
    blocks: tuple[str, ...] = field(static=True, converter=tuple)
    #: Matrix size, including ports with no blocks.
    number_of_ports: int | None = field(default=None, static=True)

    def __post_init__(self):
        if self.number_of_ports is None:
            if not self.blocks:
                raise ValueError('Empty blocks require number_of_ports.')
            self.number_of_ports = max(max(_block_indices(block)[1:]) for block in self.blocks) + 1
        if not isinstance(self.number_of_ports, int) or self.number_of_ports < 1:
            raise ValueError('number_of_ports must be a positive integer.')
        if len(set(self.blocks)) != len(self.blocks):
            raise ValueError('Port discrepancy blocks must be unique.')
        for block in self.blocks:
            _, i, j = _block_indices(block)
            if max(i, j) >= self.number_of_ports:
                raise ValueError(f'Block {block!r} exceeds number_of_ports.')
        values = jnp.asarray(unwrap(self.values))
        if jnp.iscomplexobj(values) or values.shape != (len(self.blocks), 2, unwrap(self.frequency).npoints):
            raise ValueError('values must be real with shape (n_blocks, 2, nf).')

    @classmethod
    def zeros(cls, frequency: Frequency, blocks: tuple[str, ...], *, number_of_ports: int | None = None):
        """Build zero values; infer port count from blocks unless supplied.

        Supply ``number_of_ports`` for empty blocks or uncovered higher ports.
        """
        blocks = tuple(blocks)
        return cls(jnp.zeros((len(blocks), 2, frequency.npoints)), frequency, blocks, number_of_ports)

    @unwrap_self
    def __call__(self, frequency: Frequency) -> Array:
        same_shape = frequency.f.shape == self.frequency.f.shape
        mismatch = jnp.any(frequency.f != self.frequency.f) if same_shape else True
        # Circuit discovers stamp shapes with eval_shape on a dummy grid. Keep
        # the assertion in the traced computation so only evaluation rejects it.
        mismatch = jnp.logical_or(mismatch, jnp.any(jnp.zeros_like(self.values, dtype=bool)))
        values = eqx.error_if(self.values, mismatch, 'Port discrepancy requires its own frequency grid; make a joint prediction on the new grid.')
        delta = jnp.zeros((frequency.npoints, self.number_of_ports, self.number_of_ports), dtype=jnp.result_type(values, 1j))
        for k, block in enumerate(self.blocks):
            kind, i, j = _block_indices(block)
            indices = jnp.arange(frequency.npoints)
            value = values[k, 0, indices] + 1j * values[k, 1, indices]
            delta = delta.at[:, i, j].add(value)
            if kind != 'reflection':
                delta = delta.at[:, j, i].add(value if kind == 's' else -value)
        return delta


class _CorrectedPorts(Model):
    #: Unnamed component or sub-circuit.
    model: Model
    #: Discrepancy evaluated on the component's ports.
    discrepancy: AbstractPortDiscrepancy
    #: Frozen reference impedance where the discrepancy is defined.
    z0: ArrayLike = field(converter=freeze)

    @property
    def number_of_ports(self):
        return self.model.number_of_ports

    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> Array:
        s = self.model.s(frequency, z0=self.z0)
        delta = self.discrepancy(frequency)
        if delta.shape != s.shape:
            raise ValueError(f'Port discrepancy shape {delta.shape} does not match model S shape {s.shape}.')
        diagonal = jnp.eye(self.number_of_ports, dtype=bool)
        corrected = jnp.where(diagonal, s + delta, s * jnp.exp(delta))
        return renormalize_s(corrected, self.z0, z0)


def _correct_ports(model, *, discrepancy, reference_z0):
    return _CorrectedPorts(model, discrepancy, reference_z0)


class PortCorrected(Derived):
    r"""A component or sub-circuit corrected at a reference impedance.

    The wrapper takes the component's name and stores an unnamed component,
    following :func:`pmrf.derived`. Its existing parameter names stay unchanged
    (``cable.R``), and discrepancy values become ``cable.discrepancy.values``
    inside a circuit. At the root, names are relative: ``R`` and
    ``discrepancy.values``.

    **Mathematical Formulation**

    $$\check S_{ii}=S_{ii}+\delta_{ii},\qquad
      \check S_{ij}=S_{ij}\exp(\delta_{ij})\quad(i\ne j).$$

    Corrections are applied at ``z0`` and then renormalised to the impedance
    requested from :meth:`s`.

    Parameters
    ----------
    model : Model
        Component or sub-circuit to correct.
    discrepancy : AbstractPortDiscrepancy
        Full complex discrepancy matrix on the evaluation grid.
    z0 : array_like, default=50
        Reference impedance where the discrepancy is defined.
    """

    def __init__(self, model: Model, discrepancy: AbstractPortDiscrepancy, z0: ArrayLike = 50.0, *, name: str | None = None, metadata=None):
        super().__init__(
            (_unnamed(model), {'discrepancy': _unnamed(discrepancy), 'reference_z0': freeze(z0)}),
            fn=_correct_ports, name=model.name if name is None else name, metadata=metadata,
        )


__all__ = ['AbstractPortDiscrepancy', 'GridPortDiscrepancy', 'PortCorrected']
