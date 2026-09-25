# Conductor-loss models characterised but not implemented

Reference note. It records the microstrip conductor-loss models that were evaluated and
deliberately **not** shipped, each with its physical coverage, its cost, and the specific
reason it was left out; and it carries the standing record of a contradiction between two
models that *were* shipped.

Two existing documents are assumed and **not** repeated here:

- `docs/adr/0001-line-modelling-architecture.md` — the surface-impedance normalisation
  argument, the 2.99x weight factor between Wheeler's weight and the exact slab, and why
  `RootSumSquareSlabSurfaceImpedance` is the planar default. Decision 1 there also names
  the trace/ground split as the only honest fix; that fix has since landed as
  `TraceGroundCurrentDistribution`, so the ADR's "which ParamRF does not implement" is now
  out of date on that clause alone.
- `docs/research/microstrip-loss-conventions.md` — dielectric-attenuation conventions,
  conductor loss at $t\to0$, and $Z_c$ at low frequency.

Every claim below is labelled *established* (traced to a primary source that was read),
*as-implemented* (computed against this repository), or *inferred*.

---

## Which layer each alternative would slot into

ParamRF's line modelling has five strategy roles (`CONTEXT.md`, "Line modelling"):
**Formulation**, **Dispersion**, **SurfaceImpedance**, **CurrentDistribution**, and
**Roughness**. Conductor loss is produced by the last three: a `CurrentDistribution`
returns `(SurfaceImpedance, weight)` pairs — one per conductor surface, weight in inverse
metres — and the line multiplies and sums them. This is the vocabulary used throughout
below, and it is what decides whether an alternative is even expressible.

| alternative | layer it would occupy | expressible in ParamRF? |
|---|---|---|
| Holloway–Kuester stopping distance | `CurrentDistribution` (a weight) | yes, if $\Delta(t/2\delta_{sk})$ were available in closed form |
| Rautio–Demir $n$-layer | none — it is a solver, not a strategy | no |
| Rautio–Demir two-layer | `CurrentDistribution` emitting two pairs | yes in shape; the split fractions are not closed-form |
| Rautio–Demir one-layer with $\alpha,\beta$ | `CurrentDistribution` (one pair, rescaled) | yes in shape; $\beta$ is not closed-form |
| Patel–Triverio surface admittance + CIM | replaces Formulation *and* the loss layers | no |
| Kiang–Ali–Kong full wave | replaces the whole line model | no |
| side-wall current | a $t/(t+W)$ correction on a `CurrentDistribution` weight | yes, trivially |

What ships instead, all three as peers in
`pmrf/models/components/lines/microstrip.py`:
`WheelerCurrentDistribution` (the default), `IncrementalInductanceCurrentDistribution`
and `TraceGroundCurrentDistribution`.

---

## The open question: the sign of Wheeler's error

The two non-default distributions correct the same 1942 geometry weight by different
routes and **disagree on the direction of its error**. This is unresolved. It is recorded
here rather than smoothed over, because the resolution decides which distribution should
eventually become the microstrip default — a decision deliberately deferred.

### Why the error is invisible in a fit

The conductor channel enters $Z$ only through the product $Z_s k_c$, and
$Z_s \propto \sigma^{-1/2}$. A fit with free $\sigma$ therefore absorbs *any* $k_c$ error
exactly: a 22% error in $k_c$ surfaces as $1.22^2 = 1.49\times$ copper and nothing else.
That degeneracy is why a wrong geometry weight can survive indefinitely — it never looks
like a bad $k_c$, only like an implausible conductivity. *Established, by inspection of
the model.*

### The as-implemented numbers

At $W$ = 4 mm, $H$ = 1.6 mm, $T$ = 35 um, $\varepsilon_r$ = 4.335 ($W/H$ = 2.5,
$\Re Z_c$ = 42.28 ohm), the strong-skin geometry weights are:

| distribution | weight [1/m] | vs. Wheeler 1942 | vs. the derivative |
|---|---:|---:|---:|
| `TraceGroundCurrentDistribution` | 468.39 (= 385.69 trace + 82.71 ground) | +21.4% | +47.1% |
| `WheelerCurrentDistribution` (1942 fit) | 385.69 | — | +21.2% |
| `IncrementalInductanceCurrentDistribution` | 318.34 | −17.5% | — |

*As-implemented*: these reproduce exactly against this tree
(`HammerstadJensenMicrostripFormulation`, x64 enabled). **They are larger than the 12–18%
quoted in the originating issue**, whose departure table was internally inconsistent: it
compared the split's trace-plus-ground total against a Wheeler weight evaluated at a
different $Z_c$. The as-implemented spread against Wheeler is 21.4%, not 12–18%.

### The two cases

**The trace/ground split says Wheeler is low.** Summing the physical surfaces — trace at
$2K_i/W$ plus ground at $1/W_g$, with $W_g$ from Holloway and Kuester's ground-plane
current density — lands *above* Wheeler's single bundled constant, by 21.4% as
implemented. That is inside the 12–30% band by which Holloway and Kuester report
Wheeler's rule to underpredict, which reads as partial corroboration. Its evidence is one
closed-form integration of a published current density.

**The incremental-inductance derivative says Wheeler is high.** Evaluating the recession
derivative
$k_c = Z_0^{-1}\,\partial_n Z_a(W-2n, H+2n, T-2n)|_{n=0}$
by `jax.grad` gives 318.34 against Wheeler's 385.69, a ratio of 0.825, closing to 0.958 by
$W/H = 50$. Its evidence is four independent routes landing in 306–320 for the same case:
the Hammerstad–Jensen derivative (318.3), the same derivative taken numerically on an
independent 2D quasi-static field solve that meshes the real cross-section (316.0), a
power-loss surface-impedance integral $k_c = \int\rho_s^2\,dl / Q^2$ on that same
air-filled problem (310–320), and a volumetric PEEC solve with no surface-impedance
approximation anywhere (~306). *The last three are external-solver results: they are
recorded, not reproduced, and per the chain's house rules they are deliberately not turned
into tests.*

### Which is better supported

**The incremental-inductance derivative.** Three reasons:

1. Its four routes are genuinely independent — two different $Z_a$ formulations, a
   power-loss integral, and a volumetric solve that makes no surface-impedance
   approximation at all — and they agree to within 5%. The split's case rests on a single
   analytic integration.
2. Rautio and Demir supply a third, unrelated data point with the same sign: their
   two-layer model converges to 0.07 dB (about 15%) *more* loss than their converged
   81-layer model at high frequency, and 15–30% more resistance than their one-layer
   model, and they attribute the excess to a missing physical term (side current) rather
   than to a defect in the reference. A model that sums surfaces without the terms that
   redistribute current between them tends to run high — which is structurally the same
   criticism as applies to the split. *Established* (Rautio & Demir 2003, §IV–V).
3. The split's agreement with Holloway and Kuester's 12–30% band is weaker than it looks:
   the band is a comparison against *measured* loss over lines at $Z_0$ = 50, 80 and 86
   ohm at microwave frequencies, whereas the split's excess is against Wheeler's own
   analytic weight for one 42 ohm geometry, and it shrinks to a few percent at higher
   $Z_c$. Landing inside a band is not the same as reproducing what generated it.

None of this settles the question, and neither result changes the microstrip default,
which remains `WheelerCurrentDistribution`. Wheeler's disputed level error stays **open**,
with three sources on the record: Holloway and Kuester (Wheeler low by 12–30%), Rautio and
Demir's 15%-high two-layer finding (surface-summed models run high), and the derivative's
four field solves (Wheeler high by 21%).

### What the derivative supersedes

`IncrementalInductanceCurrentDistribution` evaluates Wheeler's incremental-inductance
*rule*; `WheelerCurrentDistribution` evaluates his 1942 *closed-form fit* to it,
$k_c = (2/W)\exp[-1.2(\Re Z_c/Z_0)^{0.7}]$. The derivative **replaces that fit's
edge-crowding exponential outright** — the $2/W$ prefactor and the $K_i$ factor both
disappear into a single computed number.

The consequence for this page: **any alternative built on the 1942 fit inherits the fit's
error and is not independent evidence about it.** That includes
`TraceGroundCurrentDistribution`, whose trace term is $2K_i/W$ verbatim, so its
disagreement with the derivative is partly a disagreement about $K_i$ rather than about
the trace/ground split as such. It also rules out treating the wide-line limit $k_c\to2/W$
as a check: the derivative does not recover it (22.9 against 25.0 at $W/H = 50$), because
$2/W$ assumes a zero-thickness strip and neglects fringing.

---

## Holloway and Kuester's stopping distance (1994)

*Primary source read in full:* C. L. Holloway and E. F. Kuester, "Edge shape effects and
quasi-closed form expressions for the conductor loss of microstrip lines", *Radio Science*
29(3), 539–559, 1994, <https://doi.org/10.1029/93RS03062>. A copy is at
`docs/research/holloway1994.pdf`.

### What it is

The perturbation integral $\alpha_c \propto \oint |J_s|^2\,dl$ diverges logarithmically on
an infinitely thin strip, because $J_s\sim1/\sqrt{x}$ at the edge. Lewin
(<https://doi.org/10.1109/TMTT.1984.1132762>) and Vainshtein & Zhurav independently
proposed stopping the integral a short distance $\Delta$ before the edge, with $\Delta$ a
function of the local edge geometry only. Holloway and Kuester replace the original
high-frequency-only $\Delta$ with one obtained from a finite-element eddy-current solve of
the fields in and around the edge, valid for any $t/\delta_{sk}$ and any edge shape. This
is the **only** published route to their reported 12–30% figure: "Wheeler's rule predicts
losses between 12% and 30% lower than those obtained from both our model and from
experimental results" (§6). *Established.*

It would sit in the `CurrentDistribution` layer — it produces a weight — and is the
natural home for the ground-plane term ParamRF already takes from the same paper via
`_ground_return_effective_width`.

### Why it was not taken: where our band lands on Figure 5

$\Delta$ exists only as **Figure 5**, a raster plot of $t/\Delta$ against $t/(2\delta_{sk})$.
(The paper's body text discusses the same curves in terms of $t/\delta_{sk}$ while the
figure's abscissa is labelled $t/(2\delta_{sk})$; the placement below uses the figure's own
axis.) Read off the 90° curve, it runs from $t/\Delta = 9.2$ at dc, through a sharp
resonant peak of roughly 330 near $t/(2\delta_{sk})\approx1.4$, collapses to a plateau
around 170 by 3.5, and only creeps back toward the Lewin/Vainshtein high-frequency
asymptote of 290.8 — a value the text says "is not attained until $T/\delta_{sk}$ is of
the order of 40 or above". The peak height and its abscissa are a **digitisation read of a
raster figure**, not quoted numbers; the 9.2, the 290.8 and the 40 are quoted. *Mixed:
established for the quoted values, inferred for the peak.*

For copper at 10–500 MHz, on the figure's own axis:

| copper weight | $T$ | $t/(2\delta_{sk})$ at 10 MHz | at 500 MHz |
|---|---:|---:|---:|
| one ounce | 35 um | 0.84 | 5.92 |
| half ounce | 17.5 um | 0.42 | 2.96 |

(*As-implemented arithmetic*, $\delta_{sk}=(\pi f\mu_0\sigma)^{-1/2}$, $\sigma$ = 5.8e7
S/m: 20.90 um at 10 MHz, 2.955 um at 500 MHz.)

**The band sits directly on the resonance and its collapse.** This is the one region of
the curve where a small horizontal read error off a raster figure maps to a several-fold
error in $\Delta$, and it is also exactly the region Zhurav's closed-form fallback
provably gets wrong: the paper states Zhurav's expression does a decent job only for
$T/\delta_{sk} > 5$, does not reproduce the resonance below that, and predicts a dc value
of $t/\Delta = 160$ against the numerical 9.2. So neither the figure nor the published
closed form is usable here.

The sensitivity is soft, which is what makes this a judgement rather than an
impossibility. $\Delta$ is the cutoff on a logarithmically divergent integral, so
$\alpha\propto\ln(W/\Delta)$: a 2x error in $\Delta$ is about 7% in attenuation, a 5x
error about 18%. *Inferred* from the integral's form. Introducing ~18% of unverifiable
error in order to fix a *disputed* 12–30% defect is not a good trade in this band.

**This decision is band-specific and should be revisited for a different one.** Above
roughly 2 GHz for one-ounce copper the curve is flat and the same digitisation would be
sound. The disqualifying criterion is not "digitised, therefore unusable" — it is that the
digitisation error here is comparable to the defect being fixed. Note also the paper's own
edge-shape result: the 45° and 90° curves are essentially identical for $t/\delta_{sk}<3$
and diverge above it, so edge profile is a further term that only matters at the high end.

---

## Rautio and Demir (2003): $n$-layer, two-layer and one-layer models

*Primary source read:* J. C. Rautio and V. Demir, "Microstrip conductor loss models for
electromagnetic analysis", *IEEE Trans. MTT* 51(3), 915–921, 2003,
<https://doi.org/10.1109/TMTT.2003.808693> (author copy:
<https://www.sonnetsoftware.com/support/downloads/techdocs/MicCondLoss_Mar03.pdf>).

### Coverage and cost

The $n$-layer model divides the conductor volume into $n$ sheets inside a 2-D method-of-
moments solve and lets Maxwell's equations decide the current split, with no assumed
frequency dependence. Validated against measurement and converged at 81 layers. Its cost
is a full EM solve per frequency (34 s per frequency for the two-layer model on the
paper's hardware; the 81-layer model far more), and its accuracy depends on cell width
across the strip — the 81-layer result does not converge until the width is subdivided
into 128 cells, because side current is confined within one skin depth of the surface.
*Established.*

### Why not shipped

**The $n$-layer model has no layer to slot into.** It is a field solver, not a strategy
object: it does not produce a `(SurfaceImpedance, weight)` pair, it produces a solved
current distribution. Adopting it means adopting an EM solver, and with it the loss of the
closed-form differentiability that the fitting path depends on.

**The reduced models need coefficients that are not closed form.** The one-layer reduction
rescales the sheet impedance by fractional top- and bottom-side currents $\alpha$ and
$\beta$. The paper is explicit that "with the incorrect selection of $\alpha$ and $\beta$,
the one-layer model can be in error for loss by as much as a factor of two", that $\beta$
"can sometimes approach zero and is frequency- and geometry-dependent", and that these
factors "do not appear to have been previously considered in the literature". For the
microstrip they study, the bottom-surface current is about 1.5x the top. A ParamRF
`CurrentDistribution` could emit exactly this shape — it is one pair with a rescaled
impedance — but there is no closed form for $\beta$, so shipping it would mean shipping a
knob whose wrong setting is a 2x loss error. *Established.*

### Their transition frequencies, and why edge crowding is fully developed in our band

The paper defines three regimes separated by two transition frequencies. Below
$f_1 = R/2\pi L$ the conductor is electrically thin, current fills the volume, the edge
singularity is absent and loss is flat. Between $f_1$ and $f_2$ (where $t = 2\delta_{sk}$)
the edge singularity emerges. Above $f_2$ the classic $\sqrt f$ behaviour sets in. The
paper notes $f_1$ is "based purely on empirical observation". *Established.*

For representative ParamRF geometries, with $R=\rho/(Wt)$ and $L=Z_c/v_p$:

| geometry ($H$ = 1.6 mm, $\varepsilon_r$ = 4.335) | $\Re Z_c$ | $f_1$ | $f_2$ ($t=2\delta_{sk}$) |
|---|---:|---:|---:|
| $W$ = 1.55 mm, $T$ = 35 um | 71.6 ohm | 0.121 MHz | 14.3 MHz |
| $W$ = 4 mm, $T$ = 35 um | 42.3 ohm | 0.076 MHz | 14.3 MHz |
| $W$ = 1.55 mm, $T$ = 17.5 um | 72.0 ohm | 0.239 MHz | 57.0 MHz |

(*As-implemented arithmetic* from the paper's own definitions, evaluated against this tree
with `HammerstadJensenMicrostripFormulation`; $f_2$ depends only on $t$.)

The consequence is that **10–500 MHz is entirely above $f_1$**: the edge singularity is
fully developed across the whole band, which is why a geometry weight that omits edge
crowding is never adequate here, and why the band straddles $f_2$ rather than sitting in
one clean regime.

### A third data point on the sign question

At high frequency their two-layer model converges to 0.07 dB — about 15% — *more* loss
than the converged 81-layer model, and 15–30% more resistance than the one-layer model.
They attribute this to side current, absent from the two-layer model, whose absence lets a
stronger edge singularity form. *Established.* This is used above as independent evidence
that a surface-summed model tends to run high.

---

## Patel, Triverio and Hum: surface admittance operator with the contour integral method

*Sources:* U. R. Patel and P. Triverio, "Skin effect modeling in conductors of arbitrary
shape through a surface admittance operator and the contour integral method", *IEEE Trans.
MTT* 64(9), 2708–2717, 2016, <https://doi.org/10.1109/TMTT.2016.2593721> (preprint:
<https://arxiv.org/abs/1509.08357>); and U. R. Patel, S. V. Hum and P. Triverio, "Fast
parameter extraction for transmission lines with arbitrarily-shaped conductors and
dielectrics", *IEEE EPEPS* 2016, <https://doi.org/10.1109/EPEPS.2016.7835448>.

**Coverage.** A Dirichlet-to-Neumann surface operator relating longitudinal $E$ to
tangential $H$ on each conductor boundary, obtained by the contour integral method. It
handles genuinely arbitrary cross-sections — trapezoidal etch profiles, curved and
V-shaped edges, rough profiles — which is precisely the geometry class every closed form
on this page assumes away. It is far cheaper than volume meshing, because it discretises
only the boundary.

**Cost, and why not shipped.** It is still a numerical solve, per frequency and per
geometry, with no analytic derivative in $W$, $H$ or $T$. That is disqualifying at the
`CurrentDistribution` seam, which exists to hand the line a differentiable closed-form
weight; a solver there breaks `jax.grad` through the geometry, which the fitting and
inference paths require. Its right role in this project is as a **validation oracle** —
an independent answer to check a shipped weight against, in the same category as the field
solves quoted above — not as a formulation. *Inferred from the method's cost structure;
the papers themselves make no claim either way about differentiability.*

---

## Kiang, Ali and Kong (1991): full-wave lossy microstrip with finite thickness

*Source:* J. F. Kiang, S. M. Ali and J. A. Kong, "Modelling of lossy microstrip lines with
finite thickness", *Progress In Electromagnetics Research* PIER 4, 85–117, **1991**,
<https://doi.org/10.2528/PIER89060600>.

Listed for completeness as the full-wave reference treatment of the same problem: it
solves the actual finite-thickness lossy strip rather than perturbing a lossless solution,
so it is not subject to any of the surface-impedance or stopping-point approximations
above. It is not shippable for the same reason as the $n$-layer model — it replaces the
whole line model with a solve, and there is no layer in ParamRF's architecture that a
full-wave solver fits into.

**Note the year.** The DOI slug `PIER89060600` encodes a 1989 submission date and the
paper is frequently miscited as 1989. The publication is PIER volume 4, 1991 (confirmed
against the Crossref record for that DOI).

---

## Side-wall current

Rautio and Demir find empirically that the one-layer/32-layer resistance difference, swept
over stripline widths from 16 to 200 um, tracks $t/(t+W)$ closely, and suggest a $t/(t+W)$
modification to compensate for current on the lateral sides. *Established* (their Fig. 9
and surrounding text).

For a 35 um trace at 1.55 mm width that is $35/(35+1550)$ = **2.2%** — an order of
magnitude below the disputed 21% on the geometry weight, and below the 2–5%
surface-impedance-model difference already documented in the shipped docstrings. It is
also entirely uncharacterised at 10–500 MHz: their sweep is at 10 GHz on lines two orders
of magnitude narrower, where $t/(t+W)$ reaches 35%.

It would be trivial to express — a scalar factor on a `CurrentDistribution` weight — so
this is a priority judgement, not an architectural one. It is worth revisiting once the
sign question above is settled, since below that it is noise.

Note that `IncrementalInductanceCurrentDistribution` already includes the side walls, by
construction: its recession derivative thins the strip by $2n$ as well as narrowing it,
which is exactly why it requires a thickness-aware formulation and refuses to run without
one (pairing it with a thickness-blind formulation under-predicts by about 30%). The
$t/(t+W)$ term above is about how side current *redistributes* the edge singularity, which
is a different effect and is not captured by any weight on this page.

---

## Source list

| source | DOI / stable link |
|---|---|
| Wheeler, H. A., "Formulas for the Skin Effect", *Proc. IRE* 30(9), 412–424, 1942 | <https://doi.org/10.1109/JRPROC.1942.232015> |
| Hammerstad, E. & Jensen, O., "Accurate Models for Microstrip Computer-Aided Design", *IEEE MTT-S Digest*, 407–409, 1980 | <https://doi.org/10.1109/MWSYM.1980.1124303> |
| Lewin, L., "A Method of Avoiding the Edge Current Divergence in Perturbation Loss Calculations", *IEEE Trans. MTT* 32(7), 717–719, 1984 | <https://doi.org/10.1109/TMTT.1984.1132762> |
| Vainshtein, L. A. & Zhurav, S. M., "Strong skin effect at the edges of metal plates", *Sov. Tech. Phys. Lett.* 13(6), 298–299, 1987 | no DOI; cited via Holloway & Kuester 1994, ref. list |
| Zhurav, S. M., "Edge loss in metal plates of rectangular cross section", *Sov. Tech. Phys. Lett.* 13(3), 147–148, 1987 | no DOI; cited via Holloway & Kuester 1994, ref. list |
| Barsotti, E. L., Kuester, E. F. & Dunn, J. M., "A simple method to account for edge shape in the conductor loss in microstrip", *IEEE Trans. MTT* 39(1), 98–106, 1991 | <https://doi.org/10.1109/22.64611> |
| Holloway, C. L. & Kuester, E. F., "Edge shape effects and quasi-closed form expressions for the conductor loss of microstrip lines", *Radio Science* 29(3), 539–559, 1994 | <https://doi.org/10.1029/93RS03062> |
| Rautio, J. C. & Demir, V., "Microstrip conductor loss models for electromagnetic analysis", *IEEE Trans. MTT* 51(3), 915–921, 2003 | <https://doi.org/10.1109/TMTT.2003.808693> |
| Patel, U. R. & Triverio, P., "Skin Effect Modeling in Conductors of Arbitrary Shape Through a Surface Admittance Operator and the Contour Integral Method", *IEEE Trans. MTT* 64(9), 2708–2717, 2016 | <https://doi.org/10.1109/TMTT.2016.2593721> |
| Patel, U. R., Hum, S. V. & Triverio, P., "Fast parameter extraction for transmission lines with arbitrarily-shaped conductors and dielectrics", *IEEE EPEPS*, 2016 | <https://doi.org/10.1109/EPEPS.2016.7835448> |
| Kiang, J. F., Ali, S. M. & Kong, J. A., "Modelling of Lossy Microstrip Lines with Finite Thickness", *PIER* 4, 85–117, 1991 | <https://doi.org/10.2528/PIER89060600> |

The two Soviet-journal references are the only entries without a DOI. Both are cited here
only for facts stated *about* them in Holloway and Kuester 1994 (the 290.8 asymptote and
Zhurav's dc value of 160), which was read in full; neither was read directly.
