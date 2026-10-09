# ADR-0003: Derived models add parameters by deriving from a whole base

Status: accepted (2026-09)

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
- Each new parameter is named by its keyword; a keyword that clashes with a
  base name raises.
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
- **Reparametrising a parameter in place** (replace L by a function of new
  parameters). The relation is not a function of one parameter: it replaces
  part of the model's structure (one section becomes two), and L loses its prior.
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
