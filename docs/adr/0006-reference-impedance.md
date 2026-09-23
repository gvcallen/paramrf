# ADR-0006: Models are probed at a reference impedance; Ports define a native one

Status: accepted (2026-09)

## Context

`Model.s(freq, z0=50.0)` returns S-parameters at the reference impedance `z0`.
`Port` also carries a `z0`, but it has no effect on a circuit's S-parameters
(#186):

- The nodal and MNA solvers ignore it.
- The scattering solver uses it as the native reference of its result, then
  renormalises to the `z0` passed to `s()`, which defaults to 50.

So a circuit with a 10 Ω port reports S-parameters at 50 Ω, and `to_skrf`
labels them 50 Ω. Passing a per-port `z0` works with the nodal solver but
crashes the scattering solver. The per-port value is also used to evaluate
every internal component, which has a different number of ports.

The model is AWR's. An element such as a transmission line has no reference
impedance of its own: it is characterised by the impedance you probe it at. A
schematic's PORTs are what give its S-parameters a reference.

## Decision

### `z0` is a probe

The `z0` argument of `s()` is the reference impedance the model is probed at.
It is scalar or per-port. An explicit `z0` always wins: the result is at that
reference, renormalised if needed.

### Some models have a native reference

A model may have a **native reference impedance**. Such a model accepts
`z0=None` in `s()`, meaning "at my native reference". It sets
`supports_native_z0 = True`, a class variable on `Model` that defaults to
`False`.

Each model declares its default in the signature of its own `s()`:

- A model with a native reference defaults to `z0=None`.
- A probe-only model defaults to `z0=50.0`.

Passing `z0=None` to a probe-only model raises a clear error, naming the model
and asking for a `z0`. The raise happens where `Model` wraps the primary
methods, which is also where an explicit `z0` is converted to an array once for
every model.

For now the models with a native reference are:

- **`Port`**: its own `z0`. `Port.s(z0=None)` is therefore a matched load, S = 0.
- **`Circuit`**: the `z0` of each of its external Ports, in port order.

A Circuit built with default Ports has a native reference of 50 Ω, so its
results do not change.

### Nested circuits are probed

A parent model (a Cascade, a Circuit, a Terminated) always passes its child an
explicit `z0`. A nested Circuit is therefore probed like any other element, and
its Ports' `z0` do not matter to the parent. This matches flattening, where a
sub-circuit's Ports become virtual.

A circuit solver evaluates its internal components at a fixed internal
reference, never at the caller's `z0`. That reference is an internal detail,
cancelled by renormalising to the external ports.

### Transmission lines stay probe-only

A line could report its characteristic impedance as its native reference. It
does not, because users want a line's S-parameters at 50 Ω by default, and a
line probed at its own `Zc` has trivial reflections. A line may opt in later
under this ADR, but its default stays 50.

### `to_skrf` resolves the reference

`Model.to_skrf(frequency, z0=None)` resolves `None` to:

- the native reference, for a model that has one;
- 50 Ω, otherwise.

It passes the resolved value to both `s()` and `skrf.Network`, so
`Network.z0` always describes the data.

The model-specific parts, currently `Circuit` and `Port`, are special-cased in
one private helper beside `to_skrf`, using lazy imports. `Model` gains no hook
for them. The helper also supplies `port_names` for a Circuit, from its Ports'
`name`:

- If no Port is named, no names are set.
- Otherwise, unnamed Ports are named by their 1-based index.

`Port` is a special case for `Circuit`, and `Circuit` is a special case for
`to_skrf`. Neither implements the special case itself.

## Rejected options

- **A model-wide native reference** (every model's `s()` defaults to `None`,
  with a reference hook on `Model`). This would change every model for the sake
  of two, and blur the difference between a probe and a reference.
- **Removing `Port.z0`**, leaving the reference as a call argument only.
  Consistent, but a Port is exactly where an RF user expects to set a
  reference, and it would still be silently ignored if kept for load use.
- **Treating `z0=None` as 50 on probe-only models.** Every model would appear
  to support it, which hides the distinction this ADR is built on.
- **Raising when a Circuit with Ports is given an explicit `z0`.** Parents must
  be able to probe nested circuits.
- **`Model.port_names` or `Circuit.to_skrf` overrides.** They keep a
  conversion utility's special cases inside the models; `Model` stays lean.

## Consequences

- `Circuit.s()` and `circuit.to_skrf()` report S-parameters at the Ports'
  impedances by default. Any circuit whose Ports set a `z0` other than 50 gives
  different numbers than before, which is the fix.
- `Circuit` needs a public accessor for its external Ports in port order. It is
  the single source for both the native reference and the port names.
- A new model with a native reference must set `supports_native_z0`, default to
  `z0=None`, and be added to the `to_skrf` helper.
