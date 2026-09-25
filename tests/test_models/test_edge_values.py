r"""
Every component evaluates at zero and at every closed bound of its parameters (ADR-0007, #224).

A closed bound is part of a model's domain: a minimiser that honours bounds evaluates the
objective exactly there. So for every component exported from `pmrf.models`, each
parameter in turn is set to zero, where its validity constraint allows it, and to each
finite closed bound of that constraint. S and its derivative with respect to that
parameter must be finite, and the MNA and scattering solvers must agree on both.

Infinite bounds are closed too (Parax ADR 0001: `Positive()` is (0, inf]), but a
consumer that treats a closed bound as an evaluable point must skip non-finite ones, so
the sweep does.

Components are found by discovery. Most need constructor arguments, so each has an
entry in `EXAMPLES`, and a new component fails `test_every_component_is_swept` until it
has one.

What the sweep found, and what was done (#224):

- `Resistor.R = 0`: fixed. Its MNA stamp came from `y()`, which substituted 1e-9 ohm
  for zero, so dS/dR was 0. It now stamps as a branch, like `Inductor` (ADR-0007).
- `ShuntResistor.R = 0`: fixed. S divided by R; it is now written with R in the
  numerator (tested against scikit-rf in test_lumped.py).
- `CentreTappedTransformer.N = 0`: fixed. Its current basis divided by N; the new basis
  spans the same space and stays independent at N = 0 (test_transformers.py).
- `MicrostripLine` dielectric `ep_r = 1`: fixed. Hammerstad-Jensen's
  `sech(sqrt(ep_r - 1))` had an infinite d/dep_r at 1; it is now summed as a series
  there (test_lines.py).
- `CoupledInductors.L1 = 0`, `L2 = 0`: given an open constraint, `Positive()`.
  $M = k \sqrt{L_1 L_2}$ has an infinite derivative at zero, and is not real for a
  negative winding inductance, so negative values never worked.
- `Load.z0 = 0`, `Port.z0 = 0`: not evaluated; see `NOT_EVALUATED`.
"""
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx
import pytest

import pmrf.models
from pmrf.frequency import Frequency
from pmrf.parameters import Param
from pmrf.models import (
    Model, Circuit, Port,
    GlobalMNACircuitSolver, GlobalScatteringCircuitSolver,
    Load, Short, Open, Match, Ground, Transformer, CentreTappedTransformer,
    Autotransformer, Balun, SourceConverter, MixedModeConverter, Isolator, Splitter,
    Tee, Attenuator, Amplifier, DirectionalCoupler,
    PhaseLine, RLGCLine, PhysicalLine, DatasheetLine, CoaxialLine, MicrostripLine,
    StriplineLine, FloatingLine,
    Resistor, Capacitor, Inductor, CoupledInductors, SeriesRL, Impedance, Admittance,
    ShuntResistor, ShuntCapacitor, ShuntInductor, CapacitorQ, InductorQ,
    PiSection, TSection, LSection, BoxSection, PiSectionCLC, BoxSectionCLCC,
    TSectionLCL, LSectionLC,
)
from pmrf.models.components.lines.base import TransmissionLine

FREQ = Frequency(start=0.1, stop=1.0, npoints=3, unit='GHz')

#: One instance of each component, at ordinary values. Its parameters are what the sweep
#: moves, one at a time.
EXAMPLES = {
    Load: lambda: Load(gamma=0.3, z0=50.0),
    Short: Short,
    Open: Open,
    Match: Match,
    Port: lambda: Port(z0=50.0),
    Ground: Ground,
    Transformer: lambda: Transformer(N=2.0),
    CentreTappedTransformer: lambda: CentreTappedTransformer(N=2.0, tap=0.3),
    Autotransformer: lambda: Autotransformer(N=2.0),
    Balun: lambda: Balun(N=2.0),
    SourceConverter: SourceConverter,
    MixedModeConverter: MixedModeConverter,
    Isolator: lambda: Isolator(isolation=20.0),
    Splitter: Splitter,
    Tee: Tee,
    Attenuator: lambda: Attenuator(loss=3.0),
    Amplifier: lambda: Amplifier(gain=10.0),
    DirectionalCoupler: lambda: DirectionalCoupler(coupling=10.0),
    PhaseLine: lambda: PhaseLine(z0=50.0, theta=90.0, f0=0.5e9),
    # Not a multiple of a half wavelength on FREQ. There a nearly lossless line's Y has a
    # pole, and the MNA solver, which stamps lines from `y()`, loses precision: at
    # length=0.1 (half a wavelength at 1 GHz) with R = 0, MNA and scattering differ by 7e-6.
    RLGCLine: lambda: RLGCLine(R=0.1, L=250e-9, G=1e-6, C=100e-12, length=0.07),
    PhysicalLine: lambda: PhysicalLine(zn=50.0, ep_r=2.2, A=0.01, f_A=1e9, tand=0.001, length=1.0),
    DatasheetLine: lambda: DatasheetLine(zn=50.0, vf=0.69, k1=0.2, k2=0.01, length=1.0),
    CoaxialLine: lambda: CoaxialLine(length=0.5),
    MicrostripLine: lambda: MicrostripLine(length=0.1),
    StriplineLine: lambda: StriplineLine(length=0.1),
    FloatingLine: lambda: FloatingLine(floating=PhaseLine(z0=50.0, theta=90.0, f0=0.5e9)),
    Resistor: lambda: Resistor(R=10.0),
    Capacitor: lambda: Capacitor(C=1e-12),
    Inductor: lambda: Inductor(L=1e-9),
    CoupledInductors: lambda: CoupledInductors(L1=1e-9, L2=2e-9, k=0.5),
    SeriesRL: lambda: SeriesRL(R=10.0, L=1e-9),
    Impedance: lambda: Impedance(R=10.0, X=5.0),
    Admittance: lambda: Admittance(G=0.01, B=0.005),
    ShuntResistor: lambda: ShuntResistor(R=10.0),
    ShuntCapacitor: lambda: ShuntCapacitor(C=1e-12),
    ShuntInductor: lambda: ShuntInductor(L=1e-9),
    CapacitorQ: lambda: CapacitorQ(C=1e-12, Q=30.0),
    InductorQ: lambda: InductorQ(L=1e-9, Q=30.0),
    PiSection: lambda: PiSection(Y1=0.01, Y2=0.02, Y3=0.005),
    TSection: lambda: TSection(Z1=10.0, Z2=20.0, Z3=5.0),
    LSection: lambda: LSection(Z=10.0, Y=0.01),
    BoxSection: lambda: BoxSection(Y1=0.01, Y2=0.02, Y3=0.005, Y4=0.015),
    PiSectionCLC: lambda: PiSectionCLC(C1=1e-12, L=1e-9, C2=2e-12),
    BoxSectionCLCC: lambda: BoxSectionCLCC(C1=1e-12, L=1e-9, C2=2e-12, C3=0.5e-12),
    TSectionLCL: lambda: TSectionLCL(L1=1e-9, C=1e-12, L2=2e-9),
    LSectionLC: lambda: LSectionLC(L=1e-9, C=1e-12),
}

#: Exported component classes that are not swept, with the reason.
NOT_SWEPT = {
    TransmissionLine: "a marker base with no equations of its own",
}


#: Cases not swept, with the reason.
NOT_EVALUATED = {
    # A reference impedance of zero has no power waves, so S is undefined there. Its
    # open constraint would be `Positive()`, but z0 may be complex (ADR-0006), and
    # Parax maps a complex value through Positive's real bijector, which changes it.
    "Load.z0=0": "S is undefined at a zero reference impedance; z0 may be complex, so it cannot be Positive()",
    "Port.z0=0": "S is undefined at a zero reference impedance; z0 may be complex, so it cannot be Positive()",
}


def _components():
    """Every concrete component class exported from `pmrf.models`."""
    found = set()
    for obj in vars(pmrf.models).values():
        if (
            inspect.isclass(obj)
            and issubclass(obj, Model)
            and obj.__module__.startswith("pmrf.models.components.")
            and not getattr(obj, "__abstractmethods__", None)
        ):
            found.add(obj)
    return found


def test_every_component_is_swept():
    """A new component fails here until it has an example, so it is swept automatically."""
    assert _components() - set(NOT_SWEPT) == set(EXAMPLES)


def test_every_exclusion_is_a_case():
    """An exclusion that no longer names a case the sweep would generate is stale."""
    assert set(NOT_EVALUATED) <= {case.id for case in _all_cases()}


def _params(model):
    """Every parameter of `model`, nested ones included, as (path, Param)."""
    flat, _ = jax.tree_util.tree_flatten_with_path(model, is_leaf=lambda x: isinstance(x, Param))
    return [(path, leaf) for path, leaf in flat if isinstance(leaf, Param)]


def _getter(path):
    def get(tree):
        for key in path:
            if isinstance(key, jax.tree_util.GetAttrKey):
                tree = getattr(tree, key.name)
            elif isinstance(key, jax.tree_util.SequenceKey):
                tree = tree[key.idx]
            else:
                tree = tree[key.key]
        return tree
    return get


def _edge_values(param):
    """Zero, where the parameter's validity allows it, and its finite closed bounds."""
    validity = prx.unwrap(param.validity)
    if validity is None:
        return [0.0]
    lower, upper = (float(b) for b in validity.bounds)
    lower_closed, upper_closed = (bool(c) for c in validity.closed)
    values = []
    if lower_closed and np.isfinite(lower):
        values.append(lower)
    if upper_closed and np.isfinite(upper):
        values.append(upper)
    if not bool(validity.is_outside(jnp.asarray(0.0))) and 0.0 not in values:
        values.append(0.0)
    return values


def _all_cases():
    """Every (component, parameter, edge value), exclusions included."""
    cases = []
    for cls, make in EXAMPLES.items():
        for path, param in _params(make()):
            for value in _edge_values(param):
                name = f"{cls.__name__}{jax.tree_util.keystr(path)}={value:g}"
                cases.append(pytest.param(cls, path, value, id=name))
    return cases


def _cases():
    """The cases the sweep evaluates: every case not in `NOT_EVALUATED`."""
    return [case for case in _all_cases() if case.id not in NOT_EVALUATED]


def _at(model, path, x):
    """`model` with the parameter at `path` replaced by the declared value `x`."""
    get = _getter(path)
    param = get(model)
    physical = jnp.broadcast_to(jnp.asarray(x * param._scale), jnp.shape(param.value))
    return eqx.tree_at(get, model, physical)


def _in_circuit(model, solver):
    """`model` with a 50 ohm Port on each of its ports."""
    ports = [Port(z0=50.0) for _ in range(model.nports)]
    return Circuit([[(p, 0), (model, i)] for i, p in enumerate(ports)], solver=solver)


def _s_and_ds(cls, path, value, solver):
    model = EXAMPLES[cls]()
    s_of = lambda x: _in_circuit(_at(model, path, x), solver).s(FREQ)
    return jax.jvp(s_of, (jnp.asarray(value),), (jnp.asarray(1.0),))


# The scattering solver regularises its own system with eps = 1e-12, which moves S, and
# dS relative to its size, by about 2e-12 (see test_lumped.py). dS is compared relative
# to its largest entry, which reaches 6e6 for an inductor at zero.
SCATTERING_TOL = 1e-11


@pytest.mark.parametrize("cls, path, value", _cases())
def test_component_at_edge_value(cls, path, value):
    s, ds = _s_and_ds(cls, path, value, GlobalMNACircuitSolver())
    ref_s, ref_ds = _s_and_ds(cls, path, value, GlobalScatteringCircuitSolver())

    assert np.all(np.isfinite(s)), "non-finite S under MNA"
    assert np.all(np.isfinite(ds)), "non-finite dS under MNA"
    assert np.all(np.isfinite(ref_s)), "non-finite S under scattering"
    assert np.all(np.isfinite(ref_ds)), "non-finite dS under scattering"
    np.testing.assert_allclose(s, ref_s, rtol=0, atol=SCATTERING_TOL)
    np.testing.assert_allclose(ds, ref_ds, rtol=0, atol=SCATTERING_TOL * max(1.0, np.max(np.abs(ref_ds))))


@pytest.mark.parametrize("cls, field", [
    (Resistor, "R"), (Capacitor, "C"), (Inductor, "L"),
    (ShuntResistor, "R"), (ShuntCapacitor, "C"), (ShuntInductor, "L"),
], ids=lambda x: x if isinstance(x, str) else x.__name__)
def test_lumped_values_stay_unconstrained(cls, field):
    """Fitted equivalent circuits use negative L, C and R (ADR-0007)."""
    param = getattr(cls(**{field: -1.0}), field)
    assert param.value == -1.0
    assert param.validity is None
