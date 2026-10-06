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
  ParamRF defines it: a parameter's prior whitens it, and attaching a joint
  prior redefines it for the parameters under that prior (ADR-0005).
- **declared**: the number as written, in the units the parameter's scale
  declares (2.0 for 2 pF). Construction, bounds, priors and `Param.value` are
  in declared space.
- **physical**: the scaled, SI value (2e-12).
- **box**: the box a bounded minimiser searches, built from a parameter's
  bounds, not its prior: the unit box when both bounds are finite, declared
  space otherwise (ADR-0007).

*Avoid:* "unconstrained space" (reads as a parameter without bounds; that is
`prf.Unconstrained`), "unscaled" and "constrained" for declared space, "unit
space" (the unit hypercube; box space is the unit box only when both bounds are
finite), "base space" for box space (Parax's name for it, but ADR-0005's "base"
is a flow's latent, which is raw).

### Open and closed bounds

Each bound of a parameter is **closed** (the bound is a valid value, and a
model must evaluate there) or **open** (the bound is excluded, and a value on
it is rejected at construction). Closedness belongs to the constraint:
`pmrf.Bounded` and a `Uniform` support are closed, `Positive()` is open, and
where two constraints meet at one bound, open wins (ADR-0007).

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

### Tie and resolve

`prf.tie` derives one part of a tree from another: the **target** is removed from
the parameters and recomputed as `fn(source)` whenever the tree is resolved, so
it follows the source through `prf.update`, optimisation and sampling. Target
and source are selected by name, on either side, and a name that picks a
sub-model ties the parameters beneath it, pairing by suffix.

Two operations read a tied tree back, and they are not the same thing.

- **Resolve** (`prf.resolve`): structural. Ties are applied and every wrapper
  that only describes structure is discharged; parameters stay parameters. A
  container resolves to its own shape, so a `dict` of components comes back a
  `dict`, still parameterised.
- **Evaluation** (`prf.unwrap`): every parameter becomes its physical value.
  The result is numbers, not priors, and nothing rebuilt from it is
  parameterised. A joint prior is dropped the same way and for the same
  reason, so resolve leaves it standing.

Resolve is the one ordinary modelling code wants; `prf.unwrap` is a low-level
evaluation primitive. A tie's target resolves to a plain value either way — it
is derived, so it has no prior of its own.

*Avoid:* "unwrap" for reading a tied container; "apply the ties" for evaluation.

### Prior and joint prior

`prf.prior` attaches a prior to the parameters a selector picks (ADR-0005). A
scalar distribution gives each of them its own prior. A distribution whose
event size equals the parameters' total size (their number, when they are
scalars; an array parameter counts each of its values) is a **joint prior** over
them: a
wrapper (`prf.modules.Probabilistic`) around the unchanged tree, so every
parameter keeps its name. A joint prior replaces its parameters' own priors,
and its `space` says which space its distribution is over. Raw space for its
parameters becomes the distribution's whitened space.

*Avoid:* "covered parameters" (say "the parameters under a joint prior");
"joint target" (the old collapsing node).

### Derived model

A model computed from a **base** model and **new parameters** by a function,
`f(base, **new)`, built with `prf.derived` (ADR-0003). The base and the new
parameters are held once; the base keeps its names and each new parameter is
named by its keyword. Used to derive a more complete model from a nominal one
(a wet section, a cut) and, by nesting, to share a parameter across parts.

*Avoid:* "tie with new parameters", "shared parameter" as a separate concept.

### Parameter values

`prf.values`: a name-keyed dict of arrays in one space, the form values
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

## Discrepancy modelling

A fit in Kennedy–O'Hagan form treats an observation as $\tilde h = h(\theta) +
\delta + \varepsilon$: the model prediction, a **discrepancy** and measurement
noise. `MarginalLogLikelihood` scores it; see `pmrf.discrepancy_models` and
`pmrf.likelihoods`.

### Event space and event block

The space in which probability is defined: the prediction after the event
transform, with frequency as the last (event) axis. Every other axis is a batch
axis. One entry of that batch, for example the real part of S21, is an **event
block**. Blocks are independent: a kernel chooses the covariance *within* each
block (`AutoCrossKernel` routes by block, `SharedIndependentKernel` shares
hyperparameters across blocks) and never couples two blocks.

*Avoid:* "output" or "task" for a block; "cross-covariance" for the kernel an
`AutoCrossKernel` gives a transmission block.

### Discrepancy

$\delta$, the systematic misfit between model and reality, as opposed to noise
$\varepsilon$. Either deterministic or a `GaussianProcess` over each event block.
It is on the observable unless it is a port or internal discrepancy.

### Residual

The observation minus the model prediction, in event space: $r = \tilde h -
h(\theta)$. It contains both discrepancy and noise.

### Discrepancy prediction

The distribution of $\delta$ at new frequencies $x_B$, conditioned on residuals
at the fit frequencies $x_A$, per event block:
$\mu = K_{BA}(K_{AA}+\Sigma_n)^{-1} r$ and
$\Sigma = K_{BB} - K_{BA}(K_{AA}+\Sigma_n)^{-1}K_{AB}$. It excludes noise: it is
the discrepancy, not a predicted observation.

*Avoid:* "posterior predictive" (that includes noise); "discrepancy posterior"
(ambiguous with a posterior over hyperparameters).

### Linearisation

A Gauss–Newton approximation of a fit at its MAP, with the discrepancy's
hyperparameters held fixed: the Jacobian $J = -\partial r / \partial \theta$ of the
residual with respect to the model's free parameters, and the Fisher matrix
$F = J^\top \Sigma_D^{-1} J$ with $\Sigma_D = K + \Sigma_n$ per event block. With
the prior precision $\Sigma_0^{-1}$ it gives the linearised posterior covariance
$\Sigma_\text{post} = (F + \Sigma_0^{-1})^{-1}$, over parameter values in one space
(declared by default). Fisher matrices of independent fits add.

### Joint prediction

The Gaussian over the free parameters and the discrepancy at new frequencies
together, from a linearisation: unlike a discrepancy prediction, it carries the
parameters' uncertainty into $\delta$ and the cross-covariance between them. It is
the form in which one fit's posterior becomes the next fit's joint prior.

### Port discrepancy

A discrepancy embedded at a component's ports, on its S-parameters rather than on
an observable: $\check S_{ii} = S_{ii} + \delta_{ii}$ for reflection and
$\check S_{ij} = S_{ij} e^{\delta_{ij}}$ for transmission. A circuit containing
the corrected component carries $\delta$ to any observable or quantity of
interest, so it transfers between fits that observe the component differently.
Transmission is split into a symmetric and an antisymmetric block,
$\delta_{ij} = \delta^s_{ij} \pm \delta^a_{ij}$; a reciprocal component has no
antisymmetric block. $\delta$ is defined at a reference impedance and is learnt
at that impedance.

When the component is observed directly, in an event space of the same form
(additive for reflection, logarithmic for transmission), the port discrepancy *is*
the event-space discrepancy, and is marginalised there: this is the **reference
fit**. A later fit with the component inside a larger circuit holds $\delta$ as
explicit parameters, under the reference fit's joint prediction as their prior:
this is a **transfer fit**.

*Avoid:* "embedded discrepancy" or "embedded model error" (in the literature
these embed the error in the parameters); "stage 1" and "stage 2" (say reference
fit and transfer fit).

### Internal discrepancy

A discrepancy on a quantity inside a model rather than on its ports: for a
uniform line, its characteristic impedance, attenuation and phase constant,
$\check Z_c = Z_c e^{z}$, $\check\alpha = \alpha e^{a}$, $\check\beta = \beta e^{b}$.
The correction is log-relative and per unit length, so it is independent of the
line's length. A port discrepancy carries the whole instance's error and does
not transfer to another length; an internal discrepancy does, provided the
line's $\gamma L$ scales with its length. Choose the internal quantity by what
stays the same between the instances it must transfer across.

$\alpha$ and $\beta$ are corrected separately rather than by one complex factor
on $\gamma$, which would mix phase into loss because $\beta \gg \alpha$.

It is learnt in a reference fit either from reference internal quantities, in
an event space of the same log form, or from port observables alone, and is
carried into transfer fits as a port discrepancy is.

*Avoid:* "internal hook" (no general method interception exists; each
internal discrepancy corrects one typed quantity).

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
method, a public function or a value space. Read
`docs/adr/0005-priors-by-name.md` before changing how priors are attached or
scored, and `docs/adr/0007-edge-values.md` before changing how a
circuit solver produces S, how bounds are validated, or which space a solver
moves through.
