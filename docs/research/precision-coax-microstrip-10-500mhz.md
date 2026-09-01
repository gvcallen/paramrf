# Precision coaxial and microstrip modelling from 10–500 MHz

**Audit date:** 2026-09-01

**ParamRF revision:** `e96bb863`

**Application:** fitted instrument models for global 21 cm experiments

## Executive conclusion

The updated finite-thickness microstrip model is now a credible fitting model for
10–500 MHz. A known trace thickness gives the correct trace DC resistance,

$$R_{dc}=\frac{\rho}{Wt},$$

and ParamRF blends it smoothly into the strong-skin-effect resistance. This removes the
previously unphysical $R\rightarrow0$ limit and makes low-frequency roll-off expressible.
It does **not** make the transition an exact finite-thickness electromagnetic solution:
the root-sum-square resistance blend is a ParamRF convention, the conductor reactance is
still the semi-infinite $\sqrt f$ term, and the DC ground-return geometry is absent.
The accurate description is therefore **correct resistance endpoints, smooth and
fit-friendly transition**, not “validated exact DC-to-skin transition.”

For a precision fitted microstrip model, the best current basis is:

```python
MicrostripLine(
    ...,
    t=<known copper thickness>,
    formulation=HammerstadJensenMicrostripFormulation(),
    dispersion=KirschningJansenMicrostripDispersion(),
    dielectric=<fitted causal dielectric>,
    conductor=<fitted conductor>,
)
```

This configuration is **not** obtained from a default-constructed
`MicrostripLine`: the current default is the zero-thickness Wheeler formulation with
`t=None`, so the new DC floor is inactive. Precision use must select
Hammerstad--Jensen and provide a finite measured thickness explicitly.

Hammerstad–Jensen remains preferable to Wheeler. Kirschning–Jansen modal dispersion is
small for an ordinary 1–2 mm PCB substrate in this band, but it is not universally
negligible and is inexpensive to retain. Material dispersion, board construction,
roughness, launches, and reflections can all matter more. A causal dielectric model is
particularly important for FR-4-like materials.

The coaxial model is stronger: Tesche gives the correct solid-inner-conductor DC and
high-frequency limits and is smooth for fitting. Its intermediate-frequency conductor
impedance remains an approximation to Schelkunoff's cylindrical Bessel-function
solution, and ParamRF still treats the shield as infinitely thick. Exact Schelkunoff plus
a finite, separately specified outer conductor remains the main accuracy upgrade.

No formulation-level percentage can by itself prove adequacy for a global-signal
experiment. Reflections convert small phase and impedance errors into spectral ripple.
Adequacy has to be tested as the residual complex $S$-parameter response of the actual
line length and mismatch network against an allocated systematic-error budget.

## Evidence labels and audit method

This report distinguishes:

- **Published:** stated or derived in the cited primary paper.
- **Verified in source:** directly observed in the live ParamRF or official scikit-rf
  implementation.
- **Computed:** evaluated from revision `e96bb863` or from the stated equations for the
  representative geometries below. These are not measurements.
- **Inference:** an engineering consequence of the published equations or computations.

The audit covered the live line, material, and test sources, commits `d5dbafbc`,
`9b6ea4ae`, and `8df6ef53`, and the installed scikit-rf 2.1.0 source. On the clean
`e96bb863` revision, the full suite passed with **459 passed and 28 skipped**, and the
focused suite

```text
tests/test_models/test_lines.py
tests/test_models/test_lines_skrf_matrix.py
```

passed **77/77 tests**. Those tests are valuable equation and regression checks; they do
not turn scikit-rf into experimental ground truth.

During final report review, a concurrent uncommitted refactor changed
`pmrf/materials/properties.py` from `NamedTuple` records to Equinox modules. That change
was not part of this research task and was preserved. A fresh focused rerun against that
later worktree produced **5 failures and 84 passes** because existing conductor code and
tests still call the removed `._replace()` method. This does not alter the formulation
analysis below, but the final worktree was not green at handoff.

## What changed in the live microstrip model

### One consistent dispersed RLGC path

**Verified in source.** `MicrostripLine._resolved_quasi_static()` now supplies the same
effective permittivity and width used by `immittance()`, `ep_eff()`, and `w_eff()`.
With dispersion enabled, ParamRF applies Kirschning–Jansen to the quasi-static state and
then constructs physical per-unit-length $Z$ and $Y$:

$$Z=\frac{j\omega Z_{c,m}\sqrt{\varepsilon_e}}{c}+Z_sK_c,$$
$$Y=\frac{j\omega\sqrt{\varepsilon_e}}{Z_{c,m}c}+\sigma_dK_g.$$

The observable telegrapher quantities are then

$$Z_c=\sqrt{Z/Y},\qquad \gamma=\sqrt{ZY}.$$

This is more expressive than treating conductor loss as an attenuation added after the
modal impedance has been fixed: loss can affect both propagation and the line impedance.
It also means ParamRF's lossy $Z_c$ will deliberately differ from scikit-rf `MLine`,
whose official source builds `gamma` by adding conductor and dielectric attenuation to a
phase term and leaves its modal characteristic impedance independent of conductor loss.
The [official `MLine` source](https://github.com/scikit-rf/scikit-rf/blob/master/skrf/media/mline.py)
also warns that its conductor-loss calculation is invalid below three skin depths and
may fall below the physical DC resistance.

Planar transmission lines do not have a unique characteristic-impedance definition.
NIST compares power-voltage, power-current, and causal definitions explicitly in
[Williams et al.](https://www.nist.gov/publications/causal-characteristic-impedance-planar-transmission-lines).
Consequently, disagreement in lossy low-frequency $Z_c$ is not automatically an error;
the definition and the resulting measurable network response matter.

### Finite-thickness DC resistance floor

**Verified in source.** For finite `t`, ParamRF computes

$$R_{ac}=\Re(Z_sK_c),\qquad
R=\sqrt{R_{dc}^2+R_{ac}^2},$$

while preserving $\Im(Z_sK_c)$ unchanged. For `t=None`, no floor is applied; that input
means thickness is unknown and strong skin effect is asserted throughout the sweep.
Precision work should therefore provide a finite, physically known `t`.

The blend guarantees:

- $R\to R_{dc}$ when $R_{ac}\ll R_{dc}$;
- $R\to R_{ac}$ when $R_{ac}\gg R_{dc}$;
- a smooth derivative, useful for optimisation and inference.

Neither Hammerstad–Jensen nor Wheeler prescribes this blend. The finite-thickness
microstrip loss treatment of
[Pucel, Massé, and Hartwig](https://doi.org/10.1109/TMTT.1968.1126691)
is a more physical reference, while the exact planar-slab surface impedance has a
hyperbolic-cotangent transition. Applying a slab impedance to microstrip is itself only a
proxy because edge currents and the ground return still require the two-dimensional
field problem.

### Size of the correction in the requested band

**Computed.** For copper with $\rho=1.68\times10^{-8}\ \Omega\,\mathrm m$,
$W=3$ mm, $h=1.6$ mm, and Hammerstad–Jensen plus Kirschning–Jansen, the ratio of the new
blended resistance to the former strong-skin resistance is:

| Copper thickness | 10 MHz | 50 MHz | 100 MHz | 500 MHz |
|---:|---:|---:|---:|---:|
| 18 µm | 1.262 | 1.0576 | 1.029 | 1.0059 |
| 35 µm | 1.075 | 1.0155 | 1.0078 | 1.0016 |
| 70 µm | 1.019 | 1.0039 | 1.0019 | 1.0004 |

The $R_{dc}=R_{ac}$ crossover lies below 10 MHz in all three cases (approximately
5.9, 1.6, and 0.39 MHz respectively). Thus the update is very important to the DC
asymptote, material for 18 µm copper at the 10 MHz band edge, and modest for ordinary
35–70 µm copper over most of 10–500 MHz.

Width changes the balance. At 10 MHz in the same substrate, the resistance uplift for
18 µm copper is about 38% at $W=0.5$ mm, 31% at 1.5 mm, 26% at 3 mm, and 21% at 10 mm.
The floor therefore cannot be dismissed from frequency alone.

**Computed proxy, not ground truth.** Replacing the semi-infinite sheet impedance by a
finite-slab `coth` impedance but retaining ParamRF's same $K_c$ gives the following
comparison at 10 MHz:

- 35 µm: the slab's real sheet impedance is 0.921 times the semi-infinite value,
  whereas ParamRF's blend is 1.075 times it—about 16.8% high relative to this proxy;
- 18 µm: ParamRF is about 4.8% high relative to the proxy;
- 70 µm: ParamRF is about 1.6% high relative to the proxy.

This non-monotonic-looking comparison is exactly why correct endpoints must not be
confused with a validated transition shape. A coupon measurement or a full-wave/2-D
field solution is needed to decide which transition best represents a particular stackup.

### What low-frequency roll-off can now be expressed

With finite `t`, zero substrate conductivity, and sufficiently low frequency,
$Z\simeq R_{dc}$ and $Y\simeq j\omega C$. Therefore

$$\gamma\simeq\sqrt{j\omega R_{dc}C},\qquad
|Z_c|\simeq\sqrt{\frac{R_{dc}}{\omega C}},$$

so attenuation and phase constant scale as $\sqrt f$, while $|Z_c|\propto f^{-1/2}$
with phase approaching $-45^\circ$. **Inference:** this is the correct distributed RC
limit for the model and gives a genuine low-frequency roll-off rather than the previous
vanishing-resistance artefact.

Static substrate conductivity is represented separately as $G=\sigma_dK_g$. If both
$R$ and $G$ remain nonzero at DC, $\gamma\to\sqrt{RG}$ and
$Z_c\to\sqrt{R/G}$. This is physically distinct from a constant loss tangent and from
conductor skin effect. Keeping these mechanisms separate improves extrapolation, but
they can still be highly correlated in a short, nearly matched $S_{21}$ fit.

Remaining limitations are important:

1. Only resistance is transitioned. The reactive conductor term still follows the
   semi-infinite $(1+j)\sqrt f$ law rather than approaching a finite internal inductance
   with reactance proportional to $f$.
2. $R_{dc}=\rho/(Wt)$ represents the trace only. Finite ground-plane width, thickness,
   plating, and return-current spreading are absent.
3. The same bulk conductor and roughness law serve trace and ground. Plating stacks,
   copper-treatment asymmetry, temperature coefficients, and frequency-dependent
   permeability are not independently expressible.
4. Wheeler's incremental-inductance geometry factor is retained through the transition.
   Its strong-skin current distribution is not re-solved at low frequency.

For fitting, these are model-discrepancy terms, not reasons to discard the model. The
transition should be constrained by known thickness and resistivity, and residual freedom
should be validated against more than one line length.

## Quasi-static microstrip geometry

### Hammerstad–Jensen

**Published and verified in source.** ParamRF implements the homogeneous-impedance,
effective-permittivity, and finite-thickness width corrections from
[Hammerstad and Jensen](https://doi.org/10.1109/MWSYM.1980.1124303). The paper's quoted
fit errors are better than 0.2% for effective permittivity over
$\varepsilon_r<128$ and $0.01\le W/h\le100$, and better than 0.01–0.03% for the
homogeneous impedance over its stated width ranges.

Those are errors of an empirical closed form against the authors' reference calculation,
not total manufacturing or forward-model accuracy. The formulation does not include
solder mask, copper weave/anisotropy, finite board or ground width, enclosure sidewalls,
surface finish, launches, connectors, or cross-section variation.

### Wheeler

ParamRF's Wheeler option is a zero-thickness, simpler quasi-static approximation and
rejects finite `t`. It remains useful as a compact baseline or deliberately simplified
fit, but it cannot use the new DC floor because finite thickness is unavailable.
**Recommendation:** use Hammerstad–Jensen for the precision model. It exposes known
thickness to both the geometry correction and DC resistance, starts nearer the physical
cross-section, and reduces degeneracy among width, permittivity, length, and impedance.

### Complex permittivity caveat

ParamRF carries complex $\varepsilon_r$ through Hammerstad–Jensen and
Kirschning–Jansen. This gives a self-consistent complex RLGC model and is attractive for
fitting, but it is an analytic continuation of empirical formulas principally fitted for
real, low-loss permittivity. The cited papers do not establish sub-percent accuracy for
that complex continuation. Treat scikit-rf agreement on the same continuation as an
implementation cross-check, not independent physical validation.

## Modal versus material dispersion

### Kirschning–Jansen modal dispersion

The effective-permittivity fit is from
[Kirschning and Jansen (1982)](https://doi.org/10.1049/el:19820186); the accompanying
power-current impedance fit is from Jansen and Kirschning (1983). Its normalized
frequency is

$$f_n=f[\mathrm{GHz}]h[\mathrm{mm}].$$

The published domain commonly reported with the fit is
$1\le\varepsilon_r\le20$, $0.1\le W/h\le100$, and
$h/\lambda_0\le0.13$, with the numerical fit anchored from about
$\varepsilon_r=2.2$. ParamRF's smooth connection from $\varepsilon_r=1$ to 2.2 is its
own documented extension.

**Computed representative result.** For $W=3$ mm, $h=1.6$ mm,
$t=35$ µm, $\varepsilon_r=4.3(1-j0.02)$:

| Frequency | K–J change in $\Re(\varepsilon_{eff})$ | change in phase per metre | change in $\Re(Z_c)$ |
|---:|---:|---:|---:|
| 10 MHz | 0.00039% | 0.00004°/m | negligible |
| 100 MHz | 0.0143% | 0.0155°/m | −0.0023% |
| 500 MHz | 0.168% | 0.905°/m | −0.0217% |

This supports a narrow statement: **for this ordinary 1.6 mm geometry, K–J modal
dispersion is small through 500 MHz.** It does not support a universal statement.
$f_n$ grows directly with substrate height, and reflected ripple scales with the actual
electrical length and mismatch. Keep K–J enabled unless a complex-network residual test
demonstrates that removing it is below the experiment's systematic allocation.

### Dielectric dispersion

FR-4 is not one reproducible material. The measurements and causal wideband model in
[Djordjevic et al.](https://doi.org/10.1109/15.974647) show why a constant loss tangent
and constant real permittivity are not a precision-wideband default.

ParamRF currently offers:

| Dielectric model | Benefit for fitting | Principal risk |
|---|---|---|
| `ConstantDielectric` | few parameters; useful control model | constant loss tangent is noncausal and cannot express linked phase dispersion |
| `DjordjevicSarkar` | compact, causal, FR-4-like logarithmic trend | relaxation bounds and reference point must describe the actual laminate |
| `MultipoleDebye` | passive causal flexibility with interpretable poles | pole degeneracy without priors or adequate bandwidth |
| `ColeCole` | one broadened relaxation | may be too restrictive for a heterogeneous laminate |
| `TabulatedDielectric` | maximal interpolation freedom | causality/passivity and extrapolation are not guaranteed |

**Computed illustration, not a laminate specification.** With ParamRF's default
Djordjevic–Sarkar relaxation bounds and an input measurement
$\varepsilon_r=4.3$, $\tan\delta=0.02$ at 1 GHz, the representative microstrip's
$\Re(\varepsilon_{eff})$ is 3.403, 3.319, and 3.265 at 10, 100, and 500 MHz,
versus approximately 3.234 for the constant material. The resulting phase difference is
about 0.57°, 2.82°, and 4.27° per metre. In this example material dispersion is much
larger than K–J modal dispersion at the low and middle parts of the band.

**Recommendation:** start with Djordjevic–Sarkar when there is a credible reference
measurement and fit its material constants with priors. Escalate to a small multipole
Debye model only when multi-length data show repeatable residual structure. Do not let a
free static conductivity silently absorb missing conductor or launch physics.

## Conductor roughness and other microstrip omissions

ParamRF's `RoughConductor` uses the Hammerstad correction. It is a compact nuisance
parameter and can be valuable in a fit, but it saturates, scales the full complex surface
impedance together, and cannot represent a measured foil profile or separate treated and
smooth faces. At 10–500 MHz, skin depths are comparatively large, so ordinary submicron
roughness may be weak at the low end; several-micron electrodeposited foil can still alter
loss substantially toward 500 MHz.

The following effects are likely to dominate before a sub-percent closed-form fit error
does in a real 21 cm receiver:

- solder-mask and adhesive layers;
- finite ground and enclosure geometry;
- connector, launch, via, bend, and pad discontinuities;
- cross-section tolerances and copper etch shape;
- separate trace/ground plating and roughness;
- laminate anisotropy and weave;
- temperature dependence;
- radiation or coupling to nearby conductors.

These should generally be represented as explicit discontinuity networks, additional
material layers/models, or model discrepancy—not forced into $\varepsilon_r$, $\rho$,
or $\tan\delta$.

## Coaxial line assessment

### What ParamRF implements

**Published and verified in source.** `TescheCoaxialFormulation` implements Tesche's
equivalent circuit for the solid inner conductor,

$$Z_i=R_{dc}+\frac{Z_{hf}}{1+Z_{hf}/(j\omega L_{int})},$$

with $R_{dc}=1/(\pi a^2\sigma)$, $L_{int}=\mu/(8\pi)$, and
$Z_{hf}=Z_s/(2\pi a)$. See
[Tesche](https://doi.org/10.1109/TEMC.2006.888185). The model reaches the exact DC and
strong-skin asymptotes and is smooth and inexpensive.

The exact cylindrical conductor solution is Schelkunoff's Bessel-function field theory;
see the [original paper](https://doi.org/10.1002/j.1538-7305.1934.tb00679.x) and
[open Bell System scan](https://www.worldradiohistory.com/Archive-Bell-System-Technical-Journal/30s/Bell-1934d.o.pdf).
Tesche is an interpolation between limiting circuits, not that exact intermediate
solution.

For a homogeneous coax below higher-mode cutoff, the dominant mode is TEM: there is no
microstrip-like geometric/modal dispersion. Dielectric and conductor dispersion remain.
ParamRF's stated estimate

$$f_c\approx\frac{c}{\pi(a+b)\sqrt{\varepsilon_r\mu_r}}$$

is about 33 GHz for the representative 0.9/2.95 mm, $\varepsilon_r=2.25$ cable, safely
above 500 MHz. This conclusion is geometry-specific but decisive for that geometry.

### Tesche error against exact Schelkunoff in the installed reference

**Computed.** The installed scikit-rf 2.1.0 includes both `tesche` and
`schelkunoff` conductor models. For a 0.9 mm inner diameter, 2.95 mm shield inner
diameter, copper, $\varepsilon_r=2.25$, and $\tan\delta=10^{-3}$, ParamRF agrees with
scikit-rf's independent Tesche implementation to numerical precision. Relative to
scikit-rf's Schelkunoff option, ParamRF's attenuation is:

| Frequency | Tesche attenuation error | $|Z_c|$ implication |
|---:|---:|---:|
| 10 MHz | −6.97% | about 0.09% complex-impedance difference |
| 100 MHz | −2.30% | about 0.01% |
| 500 MHz | −0.93% | about 0.002% |

The versioned
[scikit-rf 2.1.0 coaxial source](https://scikit-rf.readthedocs.io/en/v2.1.0/_modules/skrf/media/coaxial.html)
is the source relevant to these numbers. The upstream `master` source visible during this
audit exposes a different resistance-only interpolation, illustrating why “matches
scikit-rf” must always name the version and model.

### Remaining coax limitations

- The outer shield is infinitely thick in ParamRF. Its DC resistance and internal-
  inductance transition are omitted; only its high-frequency inner-surface impedance is
  charged.
- Inner and outer conductors share one material and roughness law.
- Finite shield wall, braid coverage and transfer impedance, plating stacks, eccentricity,
  connector discontinuities, and temperature dependence are absent.
- `BulkConductor` supplies the semi-infinite surface impedance to Tesche. The equivalent
  circuit handles the solid inner rod's transition, but layered or magnetic conductors
  need richer surface-impedance models.

**Recommendation:** Tesche is suitable as the fitted default across 10–500 MHz and is
much better founded at low frequency than a bare $\sqrt f$ law. For a precision cable
loss prior, add an exact Schelkunoff formulation and separate finite inner/outer conductor
descriptions. For braided flexible coax, a fitted uniform coax model should be treated as
an effective propagation model, not a construction-exact one.

## Fitting strategy for a global 21 cm instrument

Global-signal calibration is unusually sensitive to structured reflection residuals.
As one scale example—not a universal requirement—the simulation in
[Sun et al., *Calibration Error in 21-centimeter Global Spectrum Experiments*](https://arxiv.org/abs/2405.17742)
found that reflection-coefficient magnitude errors of order $10^{-3}$ and sub-degree
phase errors could generate tens to hundreds of millikelvin of recovered-spectrum error
in their setup. This is why “only 0.17% in effective permittivity” cannot establish
negligibility by itself.

A defensible fitting programme is:

1. **Fix or tightly constrain measured geometry:** length, $W$, $h$, and $t$.
   Use Hammerstad–Jensen and finite `t`.
2. **Use a causal dielectric prior:** begin with Djordjevic–Sarkar; add only enough Debye
   poles to remove repeatable multi-length residuals.
3. **Fit conductor scale with physical priors:** resistivity and roughness should not be
   allowed to compensate freely for dielectric loss or connector loss. Include
   temperature information where available.
4. **Keep K–J enabled:** its effect is small for ordinary thin substrates, but keeping it
   avoids a preventable phase bias and costs no additional fit parameters unless a custom
   scale is introduced.
5. **Model launches/connectors separately:** use short fixtures or explicit networks.
6. **Fit at least two, preferably three, line lengths from the same construction.** Length
   differences identify uniform propagation; common residuals identify launches. A
   single matched $S_{21}$ trace cannot reliably separate $R$, dielectric loss, leakage,
   length, and mismatch.
7. **Validate outside the fit partition:** hold out frequencies, temperatures, line
   lengths, or reflection standards. Evaluate residual complex $S$-parameters and the
   propagated sky-temperature bias, not only RLGC parameter plausibility.
8. **Carry model discrepancy:** for thin copper near the band edge, include uncertainty
   for the resistance-transition shape and conductor reactance until validated against
   coupons or a field solver.

### Identifiability cautions

In a nearly matched line, conductor loss, dielectric loss, static leakage, and connector
loss can all reduce $|S_{21}|$. They become distinguishable through their frequency law,
phase, mismatch response, temperature dependence, and scaling with length. A more
expressive model helps only when the data contain those distinctions; otherwise it turns
physical parameters into interchangeable nuisance parameters.

Useful hierarchy:

1. known geometry and length;
2. compact causal dielectric plus copper resistivity;
3. roughness or one conductor-discrepancy parameter;
4. explicit launch networks;
5. additional Debye poles or transition-shape freedom only after multi-length residuals
   demand them.

## Updated recommendations, ranked

1. **Adopt H–J + K–J + finite `t` + a causal dielectric as the precision microstrip
   baseline.** The previous recommendation to first add a DC floor has been satisfied.
2. **Describe the new conductor treatment precisely:** asymptotically physical,
   differentiable, and fit-friendly; not an exact finite-thickness solution.
3. **Prioritise validation of the transition over further modal-dispersion refinement.**
   For ordinary substrates, K–J is already small below 500 MHz, while 18 µm copper can
   receive a 26% resistance correction at 10 MHz and the remaining transition-shape
   uncertainty is measurable.
4. **Prioritise real laminate characterization.** In the representative FR-4-like
   example, material dispersion produces degrees per metre of phase shift and exceeds
   K–J modal dispersion through much of the band.
5. **Add exact finite-thickness conductor physics when a forward-model precision claim is
   required:** at minimum a complex finite-slab transition as an alternative strategy,
   ideally a validated microstrip conductor formulation including edge and return
   currents.
6. **Add exact Schelkunoff and finite/separate shield conductors for coax accuracy.**
   Tesche remains an excellent fitting default, but its attenuation error is about 7% at
   10 MHz for the representative RG-58-like geometry.
7. **Base “negligible” claims on propagated residuals.** For a specified line, compute
   $\Delta\beta L$, $\Delta Z_c$, the complete mismatched $S$-parameter residual, and
   finally the inferred global-spectrum bias. Without the geometry, length, reflection
   coefficients, and error allocation, a definitive experiment-level negligibility claim
   is impossible.

## Source and implementation index

- ParamRF implementation:
  [`pmrf/models/components/lines/formulations.py`](../../pmrf/models/components/lines/formulations.py),
  [`pmrf/models/components/lines/physical.py`](../../pmrf/models/components/lines/physical.py),
  [`pmrf/materials/conductor.py`](../../pmrf/materials/conductor.py), and
  [`pmrf/materials/dielectric.py`](../../pmrf/materials/dielectric.py).
- ParamRF validation:
  [`tests/test_models/test_lines.py`](../../tests/test_models/test_lines.py) and
  [`tests/test_models/test_lines_skrf_matrix.py`](../../tests/test_models/test_lines_skrf_matrix.py).
- Hammerstad & Jensen, 1980:
  <https://doi.org/10.1109/MWSYM.1980.1124303>.
- Kirschning & Jansen, 1982:
  <https://doi.org/10.1049/el:19820186>.
- Jansen & Kirschning, 1983, *Arguments and an accurate model for the
  power-current formulation of microstrip characteristic impedance*, AEU 37(3/4),
  108--112: [bibliographic record](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=200902061502838645).
- Pucel, Massé & Hartwig, 1968:
  <https://doi.org/10.1109/TMTT.1968.1126691>.
- Djordjevic et al., 2001:
  <https://doi.org/10.1109/15.974647>.
- Tesche, 2007:
  <https://doi.org/10.1109/TEMC.2006.888185>.
- Schelkunoff, 1934:
  <https://doi.org/10.1002/j.1538-7305.1934.tb00679.x>.
- NIST planar characteristic impedance:
  <https://www.nist.gov/publications/causal-characteristic-impedance-planar-transmission-lines>.
- Official scikit-rf sources:
  <https://github.com/scikit-rf/scikit-rf/blob/master/skrf/media/mline.py> and
  <https://scikit-rf.readthedocs.io/en/v2.1.0/_modules/skrf/media/coaxial.html>.
