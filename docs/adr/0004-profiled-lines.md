# ADR-0004: Profiled lines vary parameters along a line through target-agnostic profiles

Status: accepted (2026-09)

## Context

ParamRF's line models (ADR-0001) are uniform: a `MicrostripLine` has one width,
one substrate, one conductor, and its `zc_and_gammaL` holds everywhere along it.
Real structures are not. A taper's width varies along z, a cable's permittivity
drifts with a moisture gradient, a deposited conductor's conductivity varies with
position. The quantity that varies is always an existing parameter of an existing
line, and it must keep taking part in fitting: its shape is described by a few
new parameters with their own priors, and those are what the user fits.

The repository already held a commented-out `ProfiledLine` from an earlier API:
a `line_fn` plus parallel `profile_fns` (static callables) and `profile_params`
dicts, a constructor that sniffed keyword types to decide what was a profile, and
a `method='stepped'|'riccati'` switch whose Riccati branch raised
`NotImplementedError`. It predates the parameter API (ADR-0002) and derived
models (ADR-0003).

Two further pressures shaped this. First (#45), a cascade of ten microstrip lines
with independent widths is built by a Python list comprehension, which traces
once per member and gives every member its own name set — the same batching
problem a profiled line has, at a different level. Second, users do not know what
section count a given taper needs for an accurate answer, and an under-resolved
line is not merely inaccurate: an optimiser will exploit it.

## Decision

`pmrf.models.ProfiledLine` takes a uniform line and a mapping from parameter
targets to **profiles**, and evaluates as a cascade of `n` uniform sections whose
parameters are sampled at the section midpoints.

### A profile is a shape, not a line

`AbstractProfile` is a `pmrf.Module` — parameter-aware, named, validated, but not
a `Model`, exactly as materials and the line strategy objects are. It declares one
method, `evaluate(t)`, on normalised `t`, documented as elementwise so the
container passes the whole midpoint array in one call. Its coefficients are
ordinary `prf.param` fields carrying physical units and their own constraints, so
`ExponentialProfile(start=2e-3, end=8e-3)` and
`ExponentialProfile(start=Bounded(...), end=Bounded(...))` both work with no
profile-specific machinery.

A profile knows nothing about the line, the parameter, or the family. This is the
central constraint: an exponential taper must apply to microstrip width, stripline
width or coaxial inner diameter without being rewritten. `evaluate` returns a
plain physical value, in the units of whatever it is attached to.

`t = 0` is at port 1 and `t = 1` at port 2, fixed on `AbstractProfile` and
restated on `ProfiledLine`. Reversing a taper swaps coefficients through
`prf.update`; there is no orientation flag, because a flag that silently mirrors
geometry inside a cascade is a debugging hazard.

The library ships `LinearProfile`, `ExponentialProfile` and `KlopfensteinProfile`.
Klopfenstein is the linearised-Riccati *design* solution: it defines a shape worth
simulating and is not a reference the simulator is checked against.

### Targets are dotted paths; the base tree is untouched

`ProfiledLine` holds the base line and a mapping from a **dotted path** naming a
`Param` on that base to the profile driving it. The base tree is left exactly as
built. Profiles are not substituted into it, and not declared as model fields.

Targets are exact dotted paths only — `w`, `substrate.dielectric.ep_r` — not
globs, sequences or callables. A glob matching three parameters would silently
create three independent profiles that happen to share a shape object, which is
never what the user meant.

`length` is not profilable and is rejected. It is the base's own parameter and it
is the **total** length; `ProfiledLine` divides by `n` internally and the user
never sees a per-section length. `ProfiledLine` has no `length` field, exposing
the base's as a read-only forwarding property.

A profiled target's value on the base is discarded, and the parameter is removed
from the profiled line's parameter set, replaced by the profile's coefficients.

That removal happens in the **naming layer**. The `Param` stays exactly where it
is in the base tree, because the container writes each profile's value back
through the driven field's own converter and constraint, and because the base
tree is left as built. What the container takes away is the target's *name*: it
declares the path **shadowed**, `pmrf.params` does not emit it, and naming it
raises rather than resolving. Merely fixing the parameter is not enough — it
stays visible and still resolves, so a fit aimed at it silently optimises
nothing, which is the #45 failure one level down. Shadowing is a general opt-in
that any container declares (`shadowed_param_paths`), not a `ProfiledLine`
special case inside the parameter machinery.

Passing a value explicitly for a target that is also profiled raises: silently
discarded input is how an afternoon is lost. A field default that is discarded
does not raise, because the user did not type it.

Coefficients are named under the target: `w.start`, `w.end`,
`substrate.dielectric.ep_r.start` — the name the parameter would have had, with
the coefficient below it. `ProfiledLine` names them itself rather than letting
the mapping's keys be named as dict keys, so a non-identifier path does not leak
a bracket form into parameter names. Globs such as `'w.*'` and `'*.start'` then
both do the obvious thing, and the container's internal field layout never
appears in a parameter name.

### Two ways in, one representation

The base may be given as a constructed line or as a class. A class is constructed
from the plain keywords, with a small reserved set (`n`, `extrapolate`, `name`,
`metadata`) always belonging to the container and a clear error on collision. A
user usually thinks "I want to profile a microstrip line", not "I want to profile
this particular line", and the class form lets them say so without constructing a
line whose width they are about to throw away. A base built this way is
deliberately unnamed, so parameter paths flatten to the container root.

Whichever form is used, one representation is stored: a base model plus the
target-to-profile mapping.

### Evaluation: midpoint sections, extrapolated

`ProfiledLine` is a `TransmissionLine` and an `AbstractBuilder`. `build` returns a
`pmrf.models.RepeatedCascade`, so every representation delegates for free and no
line-specific evaluation code is written twice.

`RepeatedCascade` is public in its own right: one model, a mapping from dotted
names to arrays whose leading axis is the repeat axis, and everything else shared.
`n` is static by construction, from that leading axis. It is what `ProfiledLine`
produces after evaluating profiles at the midpoints, and it is also the answer to
#45's cascade of independently-parametrised lines. The reduction itself moves into
`pmrf.rf` and is shared with `Cascade`: the ABCD domain by default, reduced with a
sequential `lax.scan`.

Sampling each profile at the section midpoints and building exact hyperbolic ABCD
sections is the exponential midpoint rule, an order-2 Magnus integrator. It is
time-symmetric, so its error expands in **even** powers of `h`, and Richardson
extrapolation of `T(n)` and `T(2n)` as `(4·T(2n) − T(n))/3` is **O(h⁴)**, not
O(h³). Extrapolation is on by default (`extrapolate=True`), costing `3n` section
builds, all vmapped. `n` defaults to 64, which holds the per-section electrical
length inside the guard below well past any realistic taper. `extrapolate=False`
is the escape for users who want the plain midpoint result.

Extrapolation is applied to the complex ABCD entries, and the per-frequency error
estimate is exposed.

### The guard is on by default and hard-errors

Two conditions are checked at evaluation: the Richardson error estimate against a
tolerance, and per-section electrical length against `βh ≲ 0.2 rad`. Both are
traced, so both use `eqx.error_if`, as the degenerate-DC guards in the line base
already do. A static `check=True` field turns them off.

The error estimate is taken relative to a matrix norm, never to `|S11|`:
normalising by a reflection coefficient makes the guard fire spuriously at every
reflection null.

Erroring rather than warning is deliberate. Silently wrong S-parameters feeding a
likelihood is the worst outcome available, and an under-resolved taper is exactly
the kind of numerical slack an optimiser finds and exploits. The accepted cost is
that a fit whose trial point wanders past the guard dies mid-run rather than being
penalised; `check=False` is the lever.

### Introspection

`ProfiledLine.at(t)` returns the base model with the profiled targets substituted
at that `t` — the same substitution the evaluator performs at the midpoints,
exposed for one position. It composes: `line.at(0.5).zc_and_gammaL(f)`, and `t`
as an array vmaps to give the whole profile at once.

## Rejected options

- **Profiles declared as model fields** (`w: Param | Profile`). A `prf.param`
  field runs an `as_param` converter, so a profile node is coerced or rejected;
  and where a foreign node is stored untouched (ADR-0003), the field's constraint
  and scale deliberately do not apply. A `MicrostripLine` whose width validation
  is silently bypassed by an intermediate shape object is worse than no feature.
- **Substituting the profile into the base tree in place**, replacing the `Param`
  at the target. Naming would fall out for free, but it is the same validation
  bypass at a different entry point.
- **A section function returning a concrete line** (`lambda t: MicrostripLine(w=...)`).
  It couples the taper definition to the line family: an exponential taper for
  microstrip width would have to be rewritten for stripline width and again for
  coaxial diameter. This is what forced profiles to be plain `f(t) -> value`.
- **Authoring through `prf.update`** — construct a line, then update targets with
  profile nodes. Unintuitive as the primary interface, and it reintroduces
  in-tree substitution.
- **A profile as a `Model`.** It has no ports and no S-parameters. `pmrf.Module`
  is the base for parameter-aware objects that are not models.
- **`ProfiledLine` as an `AbstractUniformLine`**, with `zc_and_gammaL` returning
  the midpoint value. A characteristic impedance that is silently the value at
  one position invites exactly the misuse the guard exists to prevent.
- **Relative profiles** (`w(t) = w_base × profile(t)`). It keeps the base value
  meaningful, but requires a reference value at every target and makes a
  dimensionless profile of permittivity mean something odd. Profiles are absolute.
- **A per-section `length`.** The base's `length` already carries a name, a prior
  and possibly a joint lab prior with the geometry; dividing it internally keeps
  all three and keeps a values dict from an earlier fit applicable.
- **`BatchedCascade` as the name.** "Batch" in ParamRF already means the
  parameter batch dimension (`prf.batch_axes`, `prf.sweep`), and a
  `RepeatedCascade` can itself be batched in that sense.
- **A `method` field and the Riccati path.** The old Riccati branch never ran. A
  field with one legal value is an API promise bought before it is needed, and a
  higher-order integrator would change `n` and `extrapolate` semantics too, so it
  deserves its own design rather than a pre-cut slot.
- **Automatic `n` selection.** `n` must be static under `jit` — it is a vmap axis
  and a scan length — while `Frequency.f` is a dynamic array leaf, so no rule that
  reads the frequency band can run inside `jit`. A manual `refine()` for power
  users is deferred to its own ticket.
- **Arbitrary-shape profiles** (polynomial, spline). They raise the smoothness
  question the extrapolation depends on and belong in a later ticket.

## Consequences

- The even-power error expansion, and therefore the O(h⁴) claim, requires the
  profile to be C² in `t`. This is documented on `AbstractProfile`. A deliberate
  kink is expressed by cascading two `ProfiledLine`s with `**`, not by a
  discontinuous profile.
- Periodic uniform sections have an artificial Bragg stopband at `βh = π`. The
  `βh ≲ 0.2` guard keeps evaluation far from it, but the mechanism is why the
  guard is always on rather than advisory.
- Richardson measures the *discretisation* error only. It is blind to the model
  error floor — mode conversion, radiation, quasi-TEM breakdown — so a small
  estimate is not a statement that the answer is physically right.
- The convergence-order test is the slow one in the suite: it measures O(h²)
  without extrapolation and O(h⁴) with it. Validation otherwise uses the
  exponential taper's closed form, a diffrax RK reference, and exactness at
  `n = 1`.
- `Cascade` and `RepeatedCascade` share one reduction in `pmrf.rf`; changes to the
  cascade numerics now affect both.
- The commented-out `nonuniform.py` is deleted rather than ported. Per `AGENTS.md`
  there is no back-compatibility obligation.
