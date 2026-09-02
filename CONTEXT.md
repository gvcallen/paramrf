# ParamRF domain context

Vocabulary for the transmission-line modelling layer. Terms here are the ones
the code uses; prefer them over synonyms. Each entry points at the class that
owns the maths rather than repeating it.

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

### ConductorShape

Surface impedance per square for one conductor cross-section, as the metal's
surface prefactor times a dimensionless shape factor. Pure numerics over an
evaluated `ConductorProperties` and a set of named dimensions; the
inverse-metre geometry weight that turns it into a per-unit-length impedance
is the caller's. See `pmrf.materials.conductor_shape.AbstractConductorShape`
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

`docs/adr/` records the decisions behind this layering. Read
`docs/adr/0001-line-modelling-architecture.md` before changing a strategy
interface or a default.
