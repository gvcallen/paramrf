# ADR-0003: Derived models and parameters add parameters by deriving from a base

Status: accepted (2026-09), amended (2026-09, #172)

## Context

Users constrain parameters by relations such as "this part is computed from
those". `prf.tie` (ADR-0002) covers this only when every source already exists
in the model. It fails when the relation needs a quantity that has no home.

The driving case (#167): a coaxial cable of total length L is wet for its first
w and dry for the rest. The model is `wet ** dry`, but neither section can hold
L. L must stay a parameter with its own prior, possibly a joint lab prior with
the cable's geometry and material, and must not drift as w changes. w is a new
parameter with its own prior. Several parts may share one such parameter (one
water level on both arms of a balun).

## Decision

`prf.derived` turns a function `f(base, **new)` returning a model into a
constructor of a **derived model**: a `pmrf.Model` (`pmrf.models.Derived`, an
`AbstractBuilder`) that holds the base and the new parameters once and calls
`f` on their unwrapped values whenever the model is used. `f` is a static field,
so models derived with one function share a jit cache entry.

### One concept, one decorator (#172)

The same relation appears at the level of a single field. Permittivity drifts
from its nominal value with temperature (the base is the existing `ep_r`, which
keeps its name and lab prior; `tc` is new), or the prior is on the velocity
factor while the line takes `ep_r` (no base; all inputs are new). A derived
model cannot express either cleanly: overriding a base parameter inside `f`
leaves it alive in `prf.params`, sampled and scored, with nothing using it.

`prf.derived` therefore gives a **derived node** that holds `(base, new)` once
and, when unwrapped, becomes `f(base, **new)`, whatever `f` returns. What the
node *is* depends on the base, decided when the constructor runs:

- A `pmrf.Model` base (or a wrapper over one, such as a joint prior) gives a
  `pmrf.Model`: `pmrf.models.Derived`, as above.
- Any other base — a `Param`, an array, a pytree — or no base gives
  `pmrf.models.DerivedValue`, a plain node that is not a model. **Narrowing:** a
  bare collection of models as a base no longer produces a model. Derive at the
  model that contains them instead.

A field declared with `prf.param` stores a derived value untouched, so every
model gets derived parameters for free. A derived value is not a parameter: no
declared value, scale, `.value`, fixed state or prior, and not in `prf.params`
or `prf.param_values`. The field's constraint and scale do not apply to it —
any constraint follows from its inputs' priors — and `f` sees and returns
physical values. Reading a derived value back by name is out of scope; it is
read from the unwrapped model.

Because a derived value collapses on unwrapping, taking its base and its new
parameters with it, their priors are scored from the wrapped tree
(`tree_derived_log_prob`) rather than mirrored against the unwrapped one. For
the same reason a tie must be resolved before the values below it collapse,
which is why `parax.Tie` opts out of having its descendants unwrapped first.

### The base stays whole

The derived model holds the base model itself, not parameters extracted from
it. This is what makes two things work with no extra machinery:

- **Joint priors.** A `Probabilistic` prior over the base (length with
  geometry) is still in the tree, over the same parameters, under the same
  names, and scores unchanged.
- **Sharing with no ties.** `f` uses the base's geometry in both sections; it
  is one parameter because it is stored once. No tie is needed to keep the wet
  and dry sections consistent.

### Naming rule

- The base's parameters keep exactly the names they have on the base. The
  wrapper is transparent to the name resolver, and the base's own name moves to
  the derived model (unless `name=` is given), so a container prefixes as usual.
- Each new parameter is named by its keyword, at the top level for a derived
  model and beside the field for a derived value (`dielectric.tc` for a value
  stored in `dielectric.ep_r`); the derived node itself is invisible to names.
  A keyword that clashes with a base name, with a sibling parameter or with
  another derived field's keyword at that level raises. Drift shared across
  several fields belongs in a derived model one level up.
- Nothing produced inside `f` is named: it is not in the tree.

A values dict saved from a fit of the base therefore applies unchanged to the
derived model.

### Sharing by nesting

A parameter shared by several parts is a new parameter of a derived model at
the level that owns them. Its `f` derives each part, passing the same value,
and puts them back with the multi-model `prf.update` mapping form (#168). A
derived model can be the base of another; names accumulate flat. There is no
separate "shared parameter" concept.

## Rejected options

- **A tie with new parameters** (`tie(..., new={...})` and other tie-centric
  designs). L ends up separate from the geometry and material, so no joint prior
  covers them, and the dry section needs extra ties to share the geometry.
- **A separate "add parameters" function** followed by a tie. Two steps for one
  idea, and the added parameters have no relation to the model until tied, with
  the same joint-prior problem.
- **Reparametrising a parameter in place** *for the wet-cable case* (replace L
  by a function of new parameters). That relation is not a function of one
  parameter: it replaces part of the model's structure (one section becomes
  two), and L loses its prior. Where the relation *is* a function of one field,
  reparametrising in place is exactly right, and #172 added it as a derived
  value.
- **Builder classes.** They work, but an engineer should not need a class for
  a one-off constraint, and topology code had to reach into the class.
  `prf.derived` is a builder whose class is generated from a function.

## Consequences

- `tie` is unchanged.
- `f` must be pure and its output's structure must not depend on parameter
  values, as for `AbstractBuilder.build`. It should be defined once at module
  level; a lambda made on every call recompiles.
- A non-model return is only detected when the model is used, since calling
  `f` eagerly at construction would repeat the work at every nesting level.
- A derived value is a structural part of the tree: storing one, like any
  structural update, recompiles. Changing its parameters' values does not.
