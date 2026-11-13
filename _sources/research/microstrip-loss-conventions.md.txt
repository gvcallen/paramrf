# Microstrip loss conventions: dielectric attenuation, conductor loss at t→0, and Z_c at low frequency

Research note. Sources are primary where they could be reached (vendor manuals, NIST
publications, standards-grade textbooks, library source code); paywalled IEEE/AEÜ papers
are cited but were **not** read in full and are marked as such. Every claim below is
labelled *established*, *inferred*, or *unverified*.

---

## Question 1 — Dielectric attenuation: complex permittivity vs. the filling-factor form

### The two expressions

ParamRF evaluates the Hammerstad–Jensen quasi-static model at a complex $\varepsilon_r$,
so its dielectric attenuation is the exact derivative of the model actually in use:

$$\alpha_d=\frac{\omega}{2c}\,\varepsilon_r\tan\delta\,
\frac{1}{\sqrt{\varepsilon_e}}\frac{\partial\varepsilon_e}{\partial\varepsilon_r}.$$

scikit-rf uses the classical filling-factor form
([`skrf/media/mline.py`, `analyse_loss`](https://scikit-rf.readthedocs.io/en/latest/_modules/skrf/media/mline.html)):

$$\alpha_d=\frac{\pi\,\varepsilon_r(\varepsilon_e-1)\tan\delta}
{(\varepsilon_r-1)\sqrt{\varepsilon_e}\,\lambda_0},$$

i.e. the same expression with $(\varepsilon_e-1)/(\varepsilon_r-1)$ standing in for
$\partial\varepsilon_e/\partial\varepsilon_r$.

### What the sources establish

**The derivative form is the original, and the filling-factor form is its linearised
surrogate.** Schneider's 1969 Bell System Technical Journal papers derive microstrip
dielectric loss from a perturbation of the stored electric energy, $\partial U/\partial
\varepsilon_r$, giving an *effective loss tangent* set by
$\partial\varepsilon_e/\partial\varepsilon_r$ — not by a filling ratio
(M. V. Schneider, "Dielectric Loss in Integrated Microwave Circuits", BSTJ 48(7), 1969,
<https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1969.tb01175.x>; and
"Microstrip Lines for Microwave Integrated Circuits", BSTJ 48(5), 1969, pp. 1421–1444,
full text at <https://archive.org/details/bstj48-5-1421>). The earlier loss measurement
work is Welch & Pratt, "Losses in Microstrip Transmission Systems for Integrated Microwave
Circuits" (1966). *Established for the existence of the derivative-based derivation; the
exact algebra of Schneider's equations was read via the abstract/summary, not the typeset
paper — see confidence table.*

**The filling factor is *defined* by assuming linearity.** Steer, *Microwave and RF
Design II — Transmission Lines*, §3.5 (Engineering LibreTexts,
<https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electronics/Microwave_and_RF_Design_II_-_Transmission_Lines_(Steer)/03:_Planar_Transmission_Lines/3.05:_Microstrip_Transmission_Lines>)
introduces $q$ by $\varepsilon_e = 1 + q(\varepsilon_r - 1)$ (eq. 17), with
$\tfrac12 \le q \le 1$, and then gives exactly the scikit-rf expression,
$\alpha_d=\frac{\omega}{c}\tan\delta\,\frac{\varepsilon_r(\varepsilon_e-1)}
{2\sqrt{\varepsilon_e}(\varepsilon_r-1)}$ (eq. 26). So $q=(\varepsilon_e-1)/(\varepsilon_r-1)$
is a *definition* that coincides with $\partial\varepsilon_e/\partial\varepsilon_r$ only if
$\varepsilon_e$ is affine in $\varepsilon_r$. Steer states the result "applies to any TEM
transmission line configuration" — a TEM/static-partition argument, not a Hammerstad–Jensen
one. **Established.**

**Hammerstad–Jensen is not affine in $\varepsilon_r$, but the error is ~1 %, not 3–9 %.**
Evaluated directly against scikit-rf's own `hammerstad_ab`/`hammerstad_er`:

| $u=W/h$ | $\varepsilon_r$ | $\varepsilon_e$ | $\partial\varepsilon_e/\partial\varepsilon_r$ | $(\varepsilon_e-1)/(\varepsilon_r-1)$ | surrogate/derivative |
|---|---|---|---|---|---|
| 1   | 4.3 | 3.1045 | 0.63286 | 0.63774 | 1.0077 |
| 10  | 10  | 8.5565 | 0.83754 | 0.83961 | 1.0025 |
| 0.5 | 2.2 | 1.7302 | 0.60238 | 0.60847 | 1.0101 |
| 3   | 9.8 | 7.3547 | 0.71925 | 0.72213 | 1.0040 |

**Finding that contradicts the premise as stated:** the linearity surrogate alone accounts
for only **0.25 %–1.0 %** of attenuation difference over this range, not 3–9 %. The u=10,
ε_r=10 numbers (0.8375 vs 0.8396) are confirmed, but that is a 0.25 % gap. A 3–9 %
observed gap therefore has a *different* dominant cause. Candidates, in order of likely
size, all *inferred* and worth confirming numerically before the docstring claims anything:
1. scikit-rf feeds the **dispersed** $\varepsilon_{e}(f)$ into a formula whose $q$ was
   derived quasi-statically, while ParamRF differentiates the quasi-static model and then
   disperses. These diverge as soon as Kirschning–Jansen dispersion is active.
2. ParamRF's complex $\varepsilon_r$ propagates through the *dispersion* model as well as
   the quasi-static one; scikit-rf takes `real(ep_reff_f)` for $\beta$ and adds
   $\alpha_d$ separately, so the dispersion model's own sensitivity to $\tan\delta$ is
   dropped.
3. Higher-order $\tan\delta$ terms: ParamRF's complex evaluation is exact in $\tan\delta$,
   the derivative form is first-order.

**ADS: complex permittivity is a material property applied globally; the microstrip loss
formula is not published.** From the primary manual, *Advanced Design System 2011.01 —
Distributed Components* (`ccdist.pdf`,
<https://edadownload.software.keysight.com/eedl/ads/2011_01/pdf/ccdist.pdf>):

- "About Dielectric Loss Models", p. 11: "Substrate loss is traditionally modeled by the
  frequency independent imaginary part of permittivity via the loss tangent (TanD)
  parameter." Since ADS 2009 the default is the causal Svensson/Djordjevic complex
  $\varepsilon(f)$ (refs: Svensson & Dermer, *IEEE Trans. Adv. Packaging* 24(2), 2001;
  Djordjevic et al.).
- MSUB Notes/Equations, note 8: "If the values of both DielectricLossModel and TanD are
  greater than zero then the real and the imaginary parts of the complex permittivity are
  frequency dependent. **This, as a material property, is applied regardless of whether a
  specific component calculates dielectric losses or not.**"
- MLIN Notes/Equations, note 1: "The frequency-domain analytical model uses the Hammerstad
  and Jensen formula to calculate the static impedance, Zo, and effective dielectric
  constant, Εeff. The attenuation factor, α, is calculated using the incremental
  inductance rule by Wheeler. … **Dielectric loss is also included in the loss
  calculation.**" No formula, no separate dielectric-loss reference. MLIN's reference list
  is Getsinger 1983, Hammerstad–Jensen 1980, Kirschning–Jansen 1982, Kobayashi, Yamashita,
  Wheeler 1942 — **no Schneider, no Welch & Pratt, no filling-factor source.**

So ADS documents (a) permittivity as a complex material property applied globally, and
(b) $\varepsilon_e$ computed by Hammerstad–Jensen — but it **never states** that H–J is
evaluated at the complex $\varepsilon_r$, and it **never gives** the dielectric attenuation
expression. The vendor documentation is **vague on precisely the point at issue**. The
absence of any dielectric-loss citation alongside an explicit Wheeler citation for
conductor loss is *suggestive* that dielectric loss falls out of the complex permittivity
rather than from a separate cited formula, but that is **inference, not established**.

**Corroborating secondary evidence for the ADS reading — scikit-rf's own source.**
`skrf/media/mline.py` splits behaviour by `compatibility_mode`, with the in-source comments:
"qucs use real-valued ep_r giving real-valued impedance" / "**ads use complex permittivity
giving complex impedance and effective permittivity**", and default (non-qucs) mode passes
the complex `ep_r_f` into `analyse_quasi_static` and `analyse_dispersion`. This is an
independent implementer's statement that carrying $\varepsilon_r$ complex through the
quasi-static/dispersion models *is* the ADS convention. Note the internal inconsistency:
scikit-rf then discards the resulting imaginary part (`real(self.ep_reff_f)` in `gamma`)
and adds the classical filling-factor $\alpha_d$ on top. **Established as a fact about
scikit-rf; secondary as evidence about ADS.**

**Qucs — the source scikit-rf follows.** The Qucs technical documentation, "Single
microstrip line" (<https://qucs.sourceforge.net/tech/node75.html>) gives the
filling-factor $\alpha_d$ and attributes both loss expressions to Hammerstad & Jensen.
Qucs deliberately keeps $\varepsilon_r$ real. **Established** (page read through a fetch
proxy; the equations are rendered as images, so the transcription of the exact grouping
should be treated with mild caution).

**AWR / Cadence Microwave Office: not obtainable.** `awrcorp.com/download/faq/.../Elements/MLIN.htm`
and the "Microstrip Line Models" user-guide page are behind a Cloudflare challenge (HTTP
403 to both WebFetch and curl). Search snippets say only that MLIN "includ[es] conductor
loss, dielectric loss and dispersion" and that AWR's FEM elements output "complex
characteristic impedances and complex effective dielectric constants". **No verifiable
primary statement about AWR's microstrip dielectric loss formulation was obtained.** The
"ADS/AWR convention" phrasing is therefore unsupported on the AWR half.

**Hammerstad & Jensen 1980 itself.** IEEE MTT-S Int. Microwave Symp. Digest, May 1980,
pp. 407–409 (also cited as 154–). The abstract states the paper gives "impedances,
effective dielectric constants, and **attenuation** including the effect of anisotropy",
and for single microstrip "dispersion and non-zero strip thickness"
(<https://ui.adsabs.harvard.edu/abs/1980mwsy.conf..154H/abstract>,
<https://www.semanticscholar.org/paper/2d551661d4d5207d0db3cf57d462e9421e9dccf4>).
**Not established:** whether the paper contains a *dielectric* attenuation formula, or only
the conductor-loss $K_i$/roughness terms. The paper is a 3-page digest and was not
obtainable in full text. Two data points cut against attributing $\alpha_d$ to it: ADS,
which explicitly cites H–J for $Z_0$/$\varepsilon_e$ and Wheeler for $\alpha$, cites
neither for dielectric loss; and Qucs's attribution of $\alpha_d$ to H–J may be a
convenience citation to the Hammerstad & Bekkadal *Microstrip Handbook* (ELAB report
STF44 A74169, Univ. of Trondheim, 1975), which is the usual home of these engineering
formulas and is not available online. **Unverified — flag as an open question.**

**Gupta, Garg, Bahl & Bhartia.** *Microstrip Lines and Slotlines* is the canonical home
of the filling-factor $\alpha_d$ and of the generalised incremental-inductance rule. The
book is not available in full text online and could not be read; its statement of the
derivation and validity conditions is **unverified here**. Wikipedia's article on the
incremental inductance rule quotes Gupta's generalised form (see Q2), which is consistent
but secondary.

---

## Question 2 — Conductor loss at zero / unspecified strip thickness

### What the sources establish

**The incremental-inductance rule is a sum over receded conductor surfaces, and it is
defined only for $t \gg \delta$.** Wheeler's rule
(H. A. Wheeler, "Formulas for the Skin Effect", *Proc. IRE* 30(9), 1942, pp. 412–424) in
Gupta's generalised form is
$$R_\text{skin}=\sum_m \frac{R_{sm}}{\mu_0}\frac{\partial L}{\partial n_m},$$
the derivative being "the differential change in inductance as surface $m$ is receded in
the $n_m$ direction". Its stated validity condition is that "thickness and corner radius
of the conductors should be large with respect to the skin depth", more strictly
$t \ge 4\delta$ (<https://en.wikipedia.org/wiki/Wheeler_incremental_inductance_rule>,
citing Wheeler 1942, Gupta et al., and Paul, *Analysis of Multiconductor Transmission
Lines*, 2007). Pucel, Massé & Hartwig, "Losses in Microstrip", *IEEE Trans. MTT* 16, 1968,
pp. 342–350 (plus corrigendum p. 1064) applied the rule to microstrip "taking into account
the finite thickness of the strip conductor"; secondary summaries state their expressions
are valid for conductors "at least four skin depths" thick. **Established** (the Pucel
paper itself is paywalled and was not read).

**Consequences for $t\to 0$:**
1. The sum runs over *every* conductor surface — the strip's top and bottom broad faces,
   its two side walls, and the ground plane. The broad-face terms are non-zero for any
   $t$, because $\partial L/\partial n$ for recession of the top and bottom faces does not
   vanish as $t\to0$. So the rule's own structure gives a **finite, non-zero** $\alpha_c$
   in the $t\to0$ limit at fixed $R_s$; only the two side-wall contributions vanish.
   **Inferred from the rule's definition — sound, but not a quotation.**
2. $t\to 0$ is *outside* the rule's stated domain ($t \ge 4\delta$), so the limit is
   formally undefined rather than physically zero. **Established.**
3. The closed form $\alpha_c=\frac{R_s}{Z_c W}K_i$, $K_i=e^{-1.2(Z_c/Z_0)^{0.7}}$, as
   given by Qucs (<https://qucs.sourceforge.net/tech/node75.html>) with the explicit
   validity note "$t > 3\delta$", **contains no $t$ at all**: it is the broad-face /
   thin-strip form of the rule, with side-wall recession folded into the empirical current
   distribution factor $K_i$. Its independence from $t$ is a modelling choice, not a
   statement that loss vanishes at $t=0$. **Established.**

**Verdict on the premise:** *verified*. Thickness enters the incremental-inductance rule
only through the side-wall terms and through the validity condition $t\gg\delta$;
$\alpha_c \to 0$ as $t\to0$ is not physically defensible, and no source asserts it.
The physical $t\to0$ behaviour at *fixed frequency* is that the sheet resistance
$\rho/t$ diverges, so the true loss **diverges**, not vanishes — the opposite of what a
`zeros()` return implies.

**What the tools actually do (convention, not physics):**
- scikit-rf: `analyse_loss` returns `a_conductor = zeros(f.shape)` whenever `t is None or
  t <= 0`, silently ignoring `rho`. (`skrf/media/mline.py`.) **Established.**
- **ADS does the same**, and documents it: MSUB Notes/Equations note 2, "Conductor losses
  are accounted for when Cond < 4.1×10¹⁷ S/m **and T > 10⁻⁹**" (ccdist.pdf). MSUB's
  default is `T = 0 mil`, i.e. lossless by default. So scikit-rf's behaviour matches the
  dominant vendor convention. **Established, primary.**
- `wcalc`/`mcalc` take the opposite tack and stay physical:
  "the loss equations included here assume that the conductor thickness is at least several
  skin depths thick" and "**when the skin depth is larger than the metal thickness, a dc
  solution is used for loss calculations**"
  (<https://web.mit.edu/~geda/arch/i386_rhel3/versions/20050830/html/mcalc-1.5/>,
  <https://wcalc.sourceforge.net/microstrip.html>). **Established.**

**Conflict to record:** vendor/library convention (zero loss when $t$ unspecified) and
physics (loss finite-to-divergent as $t\to0$) disagree. Both positions are defensible as
long as the choice is documented; "$\alpha_c=0$ because $t$ is unknown" is a *missing-input
policy*, not a limit.

---

## Question 3 — Characteristic impedance definition at low frequency

### What the sources establish

**There is no unique $Z_c$ for microstrip, and the choice is a definition, not a
measurement.** Williams, Alpert, Arz, Walker & Grabinski, "Causal Characteristic Impedance
of Planar Transmission Lines", NIST (public domain,
<https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=30758>) opens: "Classic waveguide
circuit theories cannot be applied in planar transmission lines because they do not have a
unique wave impedance. This has led to an animated debate in the literature over the
relative merits of various definitions." It then reviews Knorr & Tufekcioglu (1975),
Jansen (1978), Bianco et al. (1978), Getsinger (1979), Jansen & Koster (1982), Rautio
(1991). **Established, primary.**

**The power-current definition was chosen for weak frequency dependence — an explicitly
pragmatic criterion.** "In 1982, Jansen and Koster argued that the definition of
characteristic impedance with the weakest frequency dependence is best. On that basis,
Jansen and Koster recommended using the power-current definition." (NIST, ibid.) This is
the definition behind Jansen & Kirschning, "Arguments and an Accurate Model for the
Power-Current Formulation of Microstrip Characteristic Impedance", *AEÜ* 37, 1983,
pp. 108–112 — i.e. **exactly the $Z_c$ that ParamRF's Kirschning–Jansen dispersion model
produces**. It is defined from modal fields, $Z_{PI}=2P/|I|^2$ with $P$ the complex power
of the forward mode, and it is a *quasi-TEM modal* quantity, not an RLGC quantity.
**Established** (definition and rationale from NIST; the AEÜ paper itself is paywalled and
was not read).

**Is it meaningful below the quasi-TEM band?** The NIST paper's answer is indirect but
usable: the phase of $Z_{PV}$ and $Z_{PI}$ is *forced* to equal $\arg p_0$ by power
normalisation, and the causal theory adds a minimum-phase (Hilbert-transform) constraint
linking $|Z_C|$ to that phase. Definitions that violate it (e.g. the power/oxide-voltage
path) are demonstrably non-causal. The paper also shows that on lossy substrates the
various definitions agree at low frequency and diverge at high frequency. **What the
sources do *not* establish:** they say nothing about the $f\to0$ limit of the
Kirschning–Jansen *closed-form* $Z_c$, which is a curve fit whose stated validity band
(per ADS: $1\le\varepsilon_r\le20$, $0.1h\le W\le100h$) carries no lower frequency bound
and whose $f\to0$ value is by construction the quasi-static Hammerstad–Jensen $Z_0$.
**Inference:** the modal $Z_c$ held constant at low frequency is the conventional reported
quantity; the RLGC $\sqrt{Z/Y}$ with an $R\propto\sqrt f$ series term and no dc floor is a
*different* quantity, and its $|Z_c|\sim f^{-1/4}$, $\arg Z_c\to-22.5^\circ$ behaviour is
an artefact of the missing dc floor, not a property of microstrip.

**What the tools report.** scikit-rf's `z0_characteristic` is `_z_characteristic`, the
output of `analyse_dispersion` — the dispersed **modal** impedance. It is *not*
$\sqrt{Z/Y}$: $\alpha_c$ and $\alpha_d$ are added to $\gamma$ separately and never enter
$Z_c$. In ADS-compatibility mode it is complex only because $\varepsilon_r$ was complex,
never because of $R$. **Established** (source read directly). This is a direct precedent
for holding $Z_c$ at its modal value at low frequency.

### The dc-resistance floor

**Established:** a physical per-unit-length $R$ must have a dc floor $R_{dc}=\rho/(Wt)$;
the skin-effect $\sqrt f$ law only takes over above the frequency where $\delta \approx t$.
Johnson & Graham, *High-Speed Signal Propagation*, "Skin-Effect Region"
(<https://www.informit.com/articles/article.aspx?p=101149&seqNum=7>) places the region
boundary precisely where "the real part of the skin-effect resistance $R_{AC}$ equals the
dc resistance $R_{DC}$", and treats the regions separately rather than blending them.
NIST measurements on lossy silicon lines report the skin effect "increasing the resistance
per unit length of the line from its dc value by approximately a factor of 5 at 25 GHz"
(Williams et al., ibid.) — i.e. the dc value is the reference floor.

**Not established:** no source found states a *standard* smoothing rule. The two
candidates in the question, $R=\max(R_{dc},R_{skin})$ and
$R=\sqrt{R_{dc}^2+R_{skin}^2}$, are both engineering conventions; neither is prescribed by
any primary reference located here. The concrete practices actually documented are:
- ADS: no dc floor for microstrip conductor loss — loss is simply switched off below
  $T=10^{-9}$ (ccdist.pdf, MSUB note 2). *Established.*
- mcalc/wcalc: "when the skin depth is larger than the metal thickness, a dc solution is
  used" — effectively a hard switch, i.e. the $\max(\cdot)$ style. *Established.*
- Ansys circuit microstrip: the reference list cites Wheeler, Pucel et al., Schneider,
  Getsinger, Hammerstad–Jensen, but the documentation "does not specify which particular
  loss mechanisms are implemented"
  (<https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v242/en/Subsystems/Circuit/Content/Circuit/TRLMicrostripReferences.htm>).
  *Vague — do not cite as support for any convention.*
- IPC / SI literature: nothing primary located. The $\sqrt{R_{dc}^2+R_{ac}^2}$ blend is
  widely used in SI practice but I could not trace it to an owning primary source.
  **Only secondary evidence exists.**

---

## Confidence summary

| Claim | Status | Best source |
|---|---|---|
| Filling-factor $q$ is *defined* by assuming $\varepsilon_e$ affine in $\varepsilon_r$ | Established | Steer, *Microwave & RF Design II* §3.5, eqs. 17/26 |
| Original derivation of $\alpha_d$ is a $\partial\varepsilon_e/\partial\varepsilon_r$ energy perturbation | Established (algebra unread) | Schneider, BSTJ 1969 (both papers); Welch & Pratt 1966 |
| H–J $\varepsilon_e$ is non-affine in $\varepsilon_r$; surrogate error ≈0.25–1 % | Established (computed here) | scikit-rf `hammerstad_er`, direct numerics |
| The 3–9 % attenuation gap is caused by non-linearity alone | **Refuted** — non-linearity gives ≤1 % | this note's table |
| ADS models substrate loss as a complex permittivity applied as a global material property | Established | ADS 2011.01 `ccdist.pdf`, "About Dielectric Loss Models"; MSUB note 8 |
| ADS evaluates H–J *at* that complex $\varepsilon_r$ to obtain dielectric loss | **Inferred only — docs are vague** | ADS MLIN note 1 (says only "dielectric loss is also included") |
| "Carrying $\varepsilon$ complex through H–J is the ADS convention" | Secondary | scikit-rf `mline.py` in-source comments |
| Same claim for AWR | **Unverified — docs unreachable (Cloudflare 403)** | — |
| H&J 1980 contains a dielectric attenuation formula | **Unverified** | digest abstract only; Qucs attributes it there, ADS does not |
| Gupta et al.'s stated derivation/validity of the filling-factor form | **Unverified — book unavailable** | — |
| Incremental-inductance rule sums over all receded surfaces; needs $t\gtrsim4\delta$ | Established | Wheeler 1942 via Gupta's generalised form; Paul 2007 |
| $\alpha_c\to0$ as $t\to0$ is not physically defensible | Established (inference from the rule + $R_s=\rho/t$) | as above; mcalc's dc-solution fallback |
| ADS also zeroes conductor loss for $T\le10^{-9}$ | Established | ADS `ccdist.pdf`, MSUB note 2 |
| Microstrip $Z_c$ is definition-dependent; K–J uses power-current | Established | Williams et al., NIST, *Causal Characteristic Impedance of Planar Transmission Lines* |
| Reported $Z_c$ is the modal value, not $\sqrt{Z/Y}$, in scikit-rf | Established | `skrf/media/mline.py` |
| $f^{-1/4}$ / $-22.5^\circ$ low-frequency $Z_c$ is a missing-dc-floor artefact | Inferred | Johnson & Graham region boundary at $R_{AC}=R_{DC}$ |
| A *standard* $R_{dc}$/$R_{skin}$ blending rule exists | **Not established** | tools use hard switches or nothing |

---

## Implications for ParamRF

1. **Fix the docstring's provenance claim.** `MicrostripLine`'s "following the ADS/AWR
   convention" is over-stated on both halves. ADS documents complex permittivity as a
   global material property but never states that Hammerstad–Jensen is evaluated at the
   complex $\varepsilon_r$, and never publishes its dielectric-loss expression; AWR's
   documentation could not be obtained at all. Suggested wording: cite the *physics*
   (Schneider's $\partial\varepsilon_e/\partial\varepsilon_r$ perturbation, of which the
   complex evaluation is the exact form) as the justification, and mention ADS only as
   "consistent with ADS's treatment of permittivity as a complex material property
   (ADS 2011.01 Distributed Components, 'About Dielectric Loss Models')". Drop AWR.
2. **ParamRF's formulation is the better-founded one**, and that can be said plainly: the
   complex evaluation is exact for the model in use, whereas the filling-factor form is
   its first-order linearisation about $\varepsilon_r$, with $q$ defined by an affineness
   assumption that Hammerstad–Jensen does not satisfy. This is a case of "scikit-rf makes
   an approximation ParamRF does not", not of ParamRF being wrong.
3. **Re-derive the validation-matrix gap before setting tolerances.** The linearity
   argument justifies ~1 %, not 3–9 %. The remaining discrepancy needs a separate
   explanation — most likely quasi-static-vs-dispersed $\varepsilon_e$ in the $q$ factor,
   or the imaginary part that scikit-rf discards at `real(self.ep_reff_f)`. Per the repo's
   testing policy, do not widen a tolerance to cover the unexplained portion; isolate it by
   comparing with dispersion disabled first.
4. **Conductor loss at `t=None`.** ParamRF currently mirrors scikit-rf (and, as it turns
   out, ADS) in producing no conductor loss without a thickness. That matches vendor
   convention, so it is a defensible default — but the code comment "scikit-rf defines this
   empirical correction as zero" should say *why*: it is a missing-input policy, since the
   physical $t\to0$ limit at fixed $\rho$ diverges. If a thickness-free conductor loss is
   ever wanted, mcalc's precedent (fall back to a dc-sheet solution when $\delta>t$) is the
   documented alternative.
5. **Characteristic impedance.** Reporting the Kirschning–Jansen power-current modal $Z_c$,
   held at its quasi-static value as $f\to0$, is the mainstream convention and matches
   scikit-rf. If ParamRF's internal $Z=\gamma Z_c$, $Y=\gamma/Z_c$ inversion ever exposes
   an RLGC $\sqrt{Z/Y}$ as "the" characteristic impedance, the two must be kept distinct in
   the docs — they are different quantities, and the $f^{-1/4}$ low-frequency rise of the
   latter is an artefact of an $R$ model with no dc floor.
6. **If a dc floor is added**, note in the docstring that no primary source prescribes a
   blending rule; a smooth $\sqrt{R_{dc}^2+R_{skin}^2}$ is a defensible
   differentiability-motivated choice for a JAX-native library (the hard $\max$ has a kink),
   but it should be labelled as ParamRF's choice, not attributed to a reference.
