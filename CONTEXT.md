# ParamRF domain context

Vocabulary for ParamRF's domain layers. Terms here are the ones the code uses;
prefer them over synonyms. Each entry points at the class that owns the maths
rather than repeating it.

## Parameters

Vocabulary from ADR-0002.

### Param

A named, possibly bounded or probabilistic value in a model: `pmrf.Param`,
wrapping a Parax **variable** (the field `variable`). A **parameter name** is
the dotted path #133's resolver gives it (`feed_coax.dielectric.ep_r`), the key
for saved results, ties and selectors.

### Space

Where a parameter's number lives. Every surface defaults to **declared**.

- **raw**: the latent, unbounded array optimisers and samplers move through.
- **declared**: the number as written, in the units the parameter's scale
  declares (2.0 for 2 pF). Construction, bounds, priors and `Param.value` are
  in declared space.
- **physical**: the scaled, SI value (2e-12).

*Avoid:* "unconstrained space" (reads as a parameter without bounds; that is
`prf.Unconstrained`), "unscaled" and "constrained" for declared space, "unit
space" (the unit hypercube).

### Scale

The units a value is written in: physical = declared × scale. An explicit scale
on a value overrides a field's default; scales never multiply.

### Fixed and frozen

- **Fixed**: a parameter state. The parameter keeps its name and prior and is
  excluded from optimisation. Toggled with `prf.update(..., fixed=)`.
- **Frozen**: an opaque subtree (`prf.freeze`), for constant data. Name-based
  operations never act on it.

### Update

`prf.update` returns a copy of a model with the parts a **selector** picks
replaced. A selector is a parameter name, a glob over names, a sequence of
names, or a callable. A **validated update** (a name → value mapping
entry, `value=`, `fixed=`) goes through each parameter's constructor; a
**structural update** (a new node, `fn=`, or a mapping entry whose value is a
`Model`, keyed by sub-model name) bypasses validation. In a mapping the tier is
decided per entry by the value's type. `prf.replace` is the plain
dataclass field replace, not an update.

*Avoid:* "set values", "with values"; "update" for an optimiser step.

### Derived model

A model computed from a **base** model and **new parameters** by a function,
`f(base, **new)`, built with `prf.derived` (ADR-0003). The base and the new
parameters are held once; the base keeps its names and each new parameter is
named by its keyword. Used to derive a more complete model from a nominal one
(a wet section, a cut) and, by nesting, to share a parameter across parts.

*Avoid:* "tie with new parameters", "shared parameter" as a separate concept.

### Parameter values

`prf.param_values`: a name-keyed dict of arrays in one space, the form values
take when crossing ParamRF's boundary. Not flattened; a 1-D vector exists only
inside adapters that need one.

## Line modelling

A **line** (`pmrf.models.components.lines.physical`) owns geometry and material
modules, evaluates them, and delegates the physics to strategy objects. Five
strategy roles exist, each a separate field so a user can replace one without
touching the others.

### Formulation

Closed-form quasi-static physics. It produces the complete electrical state a
model needs to reach S-parameters: either a per-unit-length
`ImmittanceResult` directly (coax), or a `PlanarQuasiStaticResult` the line
converts into one (microstrip, stripline). A line has exactly one.
See `pmrf.models.components.lines.formulations`.

### Dispersion

Modifies an existing quasi-static state with modal frequency dependence. It
produces no state of its own, so it exists only where the cross-section is
inhomogeneous and the mode is therefore not strictly TEM: microstrip has one
(`KirschningJansenMicrostripDispersion`), homogeneously filled coax and
stripline do not. `None` disables it.

### SurfaceImpedance

Surface impedance per square for one conductor cross-section, as the metal's
surface prefactor times a dimensionless shape factor. Pure numerics over an
evaluated `ConductorProperties` and a set of named dimensions; the
inverse-metre geometry weight that turns it into a per-unit-length impedance
is the caller's. See `pmrf.materials.surface_impedance.AbstractSurfaceImpedance`
for the normalisation convention that couples the two.

### CurrentDistribution

How a line's surface current divides across its conductors: it returns
`(shape, weight)` pairs for a given cross-section and solved quasi-static
state. It chooses shapes and weights only — dimensions reach a shape from the
cross-section record. Each strategy is written for one planar line family and
declares it in `cross_section_type`. Coax has no such layer (see ADR-0001).
See `pmrf.models.components.lines.current_distribution`.

### Roughness

Modifies conductor behaviour rather than line state — it scales a surface
impedance — so it belongs to the material, not the geometry: it is a field of
`RoughConductor` in `pmrf.materials.conductor`.

## Records at the boundaries

These frozen records are the seams that keep `Param`s and `Module`s out of the
physics classes, so a formulation or shape can be checked against its source
paper with no ParamRF objects in sight.

- **`ConductorProperties`**, **`DielectricProperties`**
  (`pmrf.materials.properties`) — a material evaluated at a frequency. A
  conductor carries the surface prefactor `zs`, the static conductivity
  `sigma`, and `gamma(omega)`, the bulk diffusion constant, as *independent*
  inputs (ADR-0001).
- **`AbstractPlanarCrossSection`** (`MicrostripCrossSection`,
  `StriplineCrossSection`) — one typed dimensions record per planar family,
  handed to a current distribution at call time.
- **`PlanarQuasiStaticResult`** — the solved quasi-static state of a planar
  line: effective permittivity, characteristic impedance, effective width and
  shunt-conductance factor.
- **`ImmittanceResult`** — per-unit-length $(Z, Y)$, the internal currency
  between a formulation and a line. $R$, $L$, $G$, $C$ are derived views.

## Decisions

`docs/adr/` records the decisions behind these layers. Read
`docs/adr/0001-line-modelling-architecture.md` before changing a strategy
interface or a default, and `docs/adr/0002-parameter-api.md` before adding a
method, a public function or a value space.
