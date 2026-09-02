# ADR-0001: Line-modelling architecture

Status: accepted (2026-09)

## Context

Transmission-line models grew from single closed-form functions into a layered
set of strategy objects: a formulation, an optional dispersion, a conductor
shape, a current distribution, and a roughness correction on the material.
That layering forced a batch of decisions — several of them deliberate
compromises — whose reasoning lived nowhere but in review threads. This record
holds them. The vocabulary is in `CONTEXT.md`; the maths is in the class
docstrings, and is not repeated here.

## Decisions

### 1. Cross-section normalisation: the RSS blend is the planar default, and the exact slab cannot be

A conductor shape returns ohm/square, referred to the current *its own*
cross-section carries; the caller multiplies by an inverse-metre geometry
weight. For round conductors the two normalisations agree exactly:
`SchelkunoffRodShape` is referred to the total rod current and the caller's
weight is $1/2\pi a$.

The planar case has no such agreement. `HollowayKuesterSlabShape` is the exact
finite-thickness strip result, referred to the total strip current, so it wants
a weight of $1/(2W)$ — for a 1.55 mm trace, $322.6\,\mathrm{m^{-1}}$. Wheeler's
incremental-inductance weight, which the microstrip current distribution
supplies, is $963.7\,\mathrm{m^{-1}}$ for that trace: a factor of **2.99**.

That factor is physics, not a scale error. The two are different
decompositions of the same problem: Wheeler's weight is an
incremental-inductance result covering the ground plane and the edge crowding
as well as the strip, while the slab is a one-dimensional strip-diffusion
result containing neither. Under Wheeler's weight the exact slab returns
0.925 ohm/m at dc for 35 um copper on that trace where the true value is
0.3097 ohm/m. No rescaling repairs it: halving the slab to fix dc breaks the
strong-skin asymptote by the same factor.

Under a *frequency-independent* weight, therefore, exactly one entry is right
at both asymptotes — `RootSumSquareSlabShape`, which reaches into the caller's
normalisation through the optional `weight` argument to express its dc floor
in per-unit-length terms. It is the planar default for that reason, not as a
compatibility shim and not because it is the industry convention.

**Known limitation.** The blend is right at both ends and wrong in between,
and the error is in the reactance. Its internal inductance is 17.7x too large
at 100 kHz for 35 um copper. Measured against the exact slab as $t/\delta$
rises:

| Frequency | $t/\delta$ | $X$ ratio (RSS / exact) |
|---|---|---|
| 100 kHz | 0.17 | 17.7 |
| 1 MHz | 0.54 | 5.6 |
| 5 MHz | 1.20 | 2.5 |
| 10 MHz | 1.70 | 1.8 |
| 50 MHz | 3.79 | 1.01 |

So the default is accurate in resistance and in the strong-skin regime, and
its internal reactance should not be trusted below roughly $t/\delta \approx 2$.
The alternatives were: make the exact slab the default (rejected — it is
wrong at dc by 2.99x under the only weight available); rescale the slab
(rejected — fixes one asymptote and breaks the other); or split the weight
into separate strip and ground-plane terms in the Holloway–Kuester manner,
which is the only fix that is actually honest, and which ParamRF does not
implement.

### 2. The metal propagation constant is an explicit conductor property

`ConductorProperties` carries `zs` (the surface prefactor) and `gamma(omega)`
(the bulk diffusion constant) as independent inputs, with `gamma` computed
from $\sigma$ and $\mu_r$ rather than recovered from `zs` as $\sigma\zeta_c$.

The alternative — deriving one from the other — is an identity that holds only
for a smooth bulk metal. A `RoughConductor` scales `zs` by the roughness
factor $K$; recovering $\gamma$ from it would push $K$ into every $\gamma a$
and $\gamma t$ in the cross-section physics, changing the *diffusion* inside
the metal because its *surface* was treated. Roughness multiplies the
prefactor only.

### 3. A typed cross-section record per planar family

A current distribution receives a frozen `MicrostripCrossSection` or
`StriplineCrossSection` and declares which it accepts. Two alternatives were
weighed:

- **One flat record with optional fields for every family.** Rejected: no
  stable set of cross-section quantities exists across families — coplanar
  waveguide brings a gap width, suspended substrate an air gap, offset
  stripline an offset — so the record would grow a nullable field per family
  and every strategy would re-check which ones are populated.
- **Loose keyword arguments.** Rejected: pairing a strategy with the wrong
  family then fails as a missing-argument error deep in a call, instead of a
  type error at the boundary.

The comparable tools (scikit-rf, qucsator, wcalc) all use flat per-line-type
parameter sets. That is workable in a procedural API where the line type
selects the function; it does not survive strategies that are swappable
independently of the line. Adopting the typed seam let the legacy
conductor-loss path be deleted outright, closing the outstanding criterion of
#99.

### 4. Coax has no current-distribution layer

Deliberately. A current distribution exists to model how current *chooses* to
crowd across a cross-section — a fitted rule. In a coaxial line the current is
where the geometry puts it: the inner conductor's whole current on its surface
at radius $a$, the shield's return on its inner surface at radius $b$. The
weights are therefore exact constants, $1/2\pi a$ and $1/2\pi b$, folded into
the coaxial formulation, which selects shapes through `inner_shape` and
`shield_shape` fields instead. Adding a distribution layer would wrap two
constants in a strategy interface to satisfy symmetry with the planar lines,
which is contortion, not architecture.

### 5. Hammerstad–Jensen is the default microstrip formulation

This is a breaking numeric change: existing microstrip results move, and no
compatibility shim or opt-out flag was added. This ADR is the migration note.
`WheelerMicrostripFormulation` (Wheeler 1977) remains available and selectable.

The accuracy argument outweighs Wheeler's familiarity on two counts.
Hammerstad–Jensen is thickness-aware, and quotes fit errors below 0.2% in
$\varepsilon_e$ and 0.01–0.03% in $Z_L$ over its stated range. More
decisively, Kirschning–Jansen dispersion — the default `dispersion` — was
itself fitted against Hammerstad–Jensen's effective width, so the previous
default paired a zero-thickness quasi-static formulation with a dispersion
model fitted to a different one.

Two related decisions:

- **A formulation that cannot use an input declines it silently and locally.**
  Wheeler 1977 accepts a stated thickness and ignores it in $\varepsilon_e$
  and $Z_c$; the thickness still reaches the conductor-loss term. Emitting a
  warning was rejected: under `jit` it fires at trace time, so it appears once
  during compilation, not on the call the user made, and is silent on every
  subsequent evaluation.
- **The Wheeler name collides.** `WheelerMicrostripFormulation` is Wheeler
  1977, a quasi-static impedance approximation; `WheelerCurrentDistribution`
  is Wheeler 1942, the skin-effect loss rule and still the microstrip default.
  They share only an author, and changing the formulation default does not
  touch the loss rule.

### 6. Characteristic impedance is $\sqrt{Z/Y}$, not Kirschning–Jansen's modal value

Both the quasi-static and dispersed paths report $Z_c=\sqrt{Z/Y}$ from the
line's own RLGC state. With dispersion enabled, the dispersed
$(\varepsilon_e, Z_c)$ replace the quasi-static ones in a fresh
`PlanarQuasiStaticResult` before conversion, so the dispersion path does not
tautologically reproduce Kirschning–Jansen's modal $Z_c$.

The alternative — reporting K–J's value directly — was rejected because
microstrip has no unique characteristic impedance (NIST: Williams, Alpert,
Arz et al., *Causal Characteristic Impedance of Planar Transmission Lines*).
K–J's is a power-current quasi-TEM modal quantity, chosen by Jansen & Kirschning
for its weak frequency dependence, which is a good property for a fitting
target and not a reason to prefer it as the reported impedance of a circuit
element. $\sqrt{Z/Y}$ is at least the impedance the rest of the simulator
actually uses.

### 7. Complex permittivity carries dielectric loss; static conductivity stays separate

Dielectric loss is carried through the quasi-static and dispersion
formulations as a complex permittivity, so it arrives in $\varepsilon_e$ and
needs no separate attenuation term. Static bulk conductivity is *not* folded
into that permittivity: as a $\sigma/j\omega\varepsilon_0$ term it is singular
at dc. It is applied separately as $G=\sigma K_g$, using the geometric
`shunt_conductance_factor` of the quasi-static result, which keeps the shunt
conductance finite and nonzero down to dc.

This matches ADS's documented treatment of permittivity as a complex material
property, though the vendor documentation does not establish the expression
used; the complex evaluation is the exact loss perturbation of the
effective-permittivity model, following Schneider's energy-perturbation
derivation.

### 8. Unspecified thickness is not zero thickness

`t=None` means the thickness was not stated, not that it is zero. It asserts
that skin effect is in operation at every frequency, which is the good-faith
default given that the underlying $R_s$ is itself a thick-conductor result, and
it takes no dc resistance floor: there is no dc regime for a floor to describe.
A stated positive $t$ gets the floor $R_{dc}=1/(\sigma W t)$, blended with the
skin-effect term.

The alternative reading — `None` as $t=0$, hence no conductor loss — would make
an unstated dimension silently switch off physics. ADS likewise applies no
floor at all. The blend used for a stated thickness is a ParamRF convention
rather than a rule any cited source prescribes; mcalc and wcalc hard-switch to
a dc solution once skin depth exceeds thickness, an equally defensible
alternative.

## Consequences

- Microstrip numbers change with the Hammerstad–Jensen default; there is no
  opt-out beyond passing `WheelerMicrostripFormulation()` explicitly.
- Planar internal inductance below $t/\delta \approx 2$ is not trustworthy, and
  will not be until separate strip and ground-plane weights exist.
- A new planar family needs a cross-section record, not a new optional field.
- A new conductor treatment scales `zs` and leaves `gamma` alone.
