# Characterised conductor-loss alternatives

This page records conductor-loss models that were evaluated but deliberately not
shipped.  ParamRF separates four layers: a conductor material supplies
$\sigma$ and $\mu$; a conductor cross-section converts those properties into a
surface impedance; a current-distribution strategy assigns surface impedances and
geometry weights to trace and return paths; and a line formulation assembles the
per-unit-length immittance.  The layer named for each alternative is therefore the
place where a future implementation belongs.

The numerical baselines below cover copper with
$\rho=1.68\times10^{-8}\ \Omega\,\mathrm m$ over 10--500 MHz unless stated
otherwise.  They are calculations from the cited primary equations, not experimental
measurements.

## Holloway--Kuester stopping distance

**Layer:** current distribution. **Coverage:** finite-thickness microstrip edge
crowding, including arbitrary edge shape. **Cost:** a cheap attenuation integral once
the stopping distance $\Delta$ is known, but $\Delta$ itself comes from a finite-element
eddy-current solve.  Holloway and Kuester report Wheeler's incremental-inductance rule
as 12--30% below their measurements.  Their remedy terminates a logarithmically
singular edge-current integral at $\Delta$; the needed $t/\Delta$ is supplied only as
their Figure 5, not as a reproducible formula
([Holloway and Kuester, 1994](https://doi.org/10.1029/93RS03062)).

Figure 5 runs from $t/\Delta=9.2$ at DC toward the Lewin--Vainshtein asymptote
290.8.  Its difficult part is exactly the ParamRF band: the curve rises sharply to
about 330 at $t/\delta\approx1.5$, then falls to about 170 by
$t/\delta\approx3.5$.  One-ounce copper spans $t/\delta=0.84$--5.9 and half-ounce
copper spans 0.42--2.96 from 10 to 500 MHz.  Zhurav's closed form is not a safe
substitute: the paper says it fails for $t/\delta<5$ and gives 160 rather than 9.2
at DC ([Holloway and Kuester, 1994](https://doi.org/10.1029/93RS03062)).

This model was not shipped because digitising the near-vertical resonance and collapse
would make a small horizontal reading error a several-fold error in $\Delta$.  The
integral softens that uncertainty only logarithmically: for the representative widths,
a factor-of-two error in $\Delta$ changes attenuation by about 7%, and a factor of five
by about 18%.  That is too much unverifiable error to resolve a disputed 12--30% level
error.  Above roughly 2 GHz the curve is flat enough that digitisation could become a
defensible implementation choice; the decision is band-specific, not a blanket
rejection of digitised data ([Holloway and Kuester, 1994](https://doi.org/10.1029/93RS03062)).

The 12--30% Wheeler deficit remains an open question.  The shipped trace/ground split
lands 13.6--18.5% above Wheeler for representative 48--50 $\Omega$ lines but only
3.5--5.5% above it near 96 $\Omega$; this is partial corroboration, not closure.
Rautio and Demir independently find their two-layer approximation about 15% *above* a
converged 81-layer solve.  The literature therefore disagrees even on the sign of a
tens-of-percent correction
([Holloway and Kuester, 1994](https://doi.org/10.1029/93RS03062);
[Rautio and Demir, 2003](https://doi.org/10.1109/TMTT.2003.808693)).

## Rautio--Demir layer models

**Layer:** current distribution coupled to a field-solver formulation.
**Coverage:** finite conductor thickness, skin effect, top/bottom current transfer,
edge crowding, and side-wall current. **Cost:** sheets must be placed in a two-dimensional
electromagnetic solve that determines their current split; the reference solution used
81 layers, and the paper validates the models both against measurement and against that
converged solve.  The one-layer reduction requires frequency- and geometry-dependent
fractional top and bottom currents, and a wrong split can cause a factor-of-two loss
error ([Rautio and Demir, 2003](https://doi.org/10.1109/TMTT.2003.808693)).

For the paper's representative geometries, the current-crowding transition
$f_1=R/(2\pi L)$ is 0.18--0.6 MHz, below this audit band.  The thickness transition
$f_2$, defined by $t=2\delta$, is 14 MHz for one-ounce copper and 57 MHz for
half-ounce copper.  Thus edge crowding is established across 10--500 MHz while
through-thickness diffusion crosses inside it.  These models were not shipped because
they are field-solver constructs rather than closed-form, differentiable strategies;
they remain strong validation oracles
([Rautio and Demir, 2003](https://doi.org/10.1109/TMTT.2003.808693)).

### Side-wall current

**Layer:** current distribution. **Coverage:** current on the vertical walls of a
finite-thickness trace, omitted by sheet-only trace models. **Cost:** an additional
side-wall current term or cross-section field solve rather than another material or
surface-impedance evaluation.  Rautio and Demir attribute the residual approximately to
$t/(t+W)$, about 2.2% for $t=35\ \mu$m and $W=1.55$ mm.  It was not shipped because
that is smaller than the transition errors below and was not validated against
10--500 MHz data; a future arbitrary-cross-section current-distribution entry could
charge it explicitly
([Rautio and Demir, 2003](https://doi.org/10.1109/TMTT.2003.808693)).

## Surface-admittance contour integral

**Layer:** conductor cross-section plus a numerical line formulation.
**Coverage:** arbitrary cross-sections, including trapezoidal, curved, and V-shaped
conductors, without volume meshing. **Cost:** a contour discretisation and linear solve
for every frequency and geometry, with no analytic derivative supplied by the method.
It was not shipped because that cost and derivative boundary do not fit a JAX-native
closed-form formulation.  It is a useful high-fidelity validation oracle for etched or
otherwise non-rectangular conductors
([Patel and Triverio, 2016](https://doi.org/10.1109/TMTT.2016.2593721)).

## Kiang--Ali--Kong finite-thickness microstrip

**Layer:** full line formulation. **Coverage:** a full-wave treatment of lossy
microstrip with finite conductor thickness. **Cost:** a numerical full-wave solution,
not a modular surface-impedance or current-distribution component.  It was not shipped
because it would replace the analytic line formulation rather than deepen one of its
reusable layers; it remains a completeness and validation reference.  The publication
year is 1991, despite the 1989-looking DOI slug
([Kiang, Ali, and Kong, 1991](https://doi.org/10.2528/PIER89060600)).

## Audit baselines

### Root-sum-square blend against the exact slab

The table uses 35 $\mu$m copper.  Values are series resistance in $\Omega$/m at
10 and 50 MHz.  “Exact slab” is the total-current half-thickness
$\zeta_c\coth(\gamma t/2)$ result; “half-space” is $\zeta_c$; and “blend” is the
named ParamRF compatibility convention.  The cross-section entries are charged with
the same current-distribution weight, so the table isolates transition shape
([Holloway and Kuester, 1994, eqs. 45 and 100](https://doi.org/10.1029/93RS03062);
[Rautio and Demir, 2003, eq. 4](https://doi.org/10.1109/TMTT.2003.808693)).

| $W$ / $h$ / modal $Z_c$ | Frequency | Exact slab | Half-space | Blend | Blend error |
|---|---:|---:|---:|---:|---:|
| 1.55 / 0.80 mm / 50 $\Omega$ | 10 MHz | 0.9669 | 0.7848 | 0.8437 | +11.5% attenuation |
|  | 50 MHz | 1.6463 | 1.7549 | 1.7820 | +0.7% attenuation |
| 0.45 / 0.25 mm / 50 $\Omega$ | 10 MHz | 3.3304 | 2.7032 | 2.9061 | +14.5% attenuation |
|  | 50 MHz | 5.6706 | 6.0446 | 6.1380 | +1.1% attenuation |
| 0.35 / 0.80 mm / 96 $\Omega$ | 10 MHz | 3.6163 | 2.9353 | 3.2399 | +16.3% attenuation |
|  | 50 MHz | 6.1575 | 6.5636 | 6.7053 | +1.4% attenuation |

After removal of each case's best global scale, the blend's maximum shape residual is
11%, 14%, and 15%, respectively.  For one-ounce copper at 10 MHz, the plain
half-space result is 5.9% high against the slab while the blend is 11.5% high; correct
endpoints do not guarantee a better transition
([Holloway and Kuester, 1994](https://doi.org/10.1029/93RS03062)).

### Tesche against Schelkunoff

For a homogeneous PTFE-filled coax with 0.9 mm inner-conductor diameter, 2.95 mm
shield inner diameter, copper conductors, $\varepsilon_r=2.25$, and
$\tan\delta=10^{-3}$, Tesche's equivalent circuit underpredicts attenuation relative
to Schelkunoff's cylindrical field solution while barely moving $|Z_c|$
([Tesche, 2007](https://doi.org/10.1109/TEMC.2006.888185);
[Schelkunoff, 1934](https://doi.org/10.1002/j.1538-7305.1934.tb00679.x)).

| Frequency | Tesche attenuation error | $|Z_c|$ difference |
|---:|---:|---:|
| 10 MHz | -6.97% | about 0.09% |
| 100 MHz | -2.30% | about 0.01% |
| 500 MHz | -0.93% | about 0.002% |

### Planar shield approximation against the exact tube

Each cell is the ratio at 10 MHz $\rightarrow$ 500 MHz to Schelkunoff's exact finite
tube.  The conductor is copper.  The planar finite-slab $\coth$ multiplied by the
infinite-wall $K_0(\gamma b)/K_1(\gamma b)$ curvature factor stays within 0.2% here
([Schelkunoff, 1934, eq. 74](https://doi.org/10.1002/j.1538-7305.1934.tb00679.x)).

| Shield inner radius $b$, wall $t$ | Planar $\coth$ | $\coth\times K_0/K_1$ | Infinite-wall $Z_{hf}$ |
|---|---:|---:|---:|
| 1.475 mm, 150 $\mu$m | 1.007 $\rightarrow$ 1.001 | 1.000 | 1.007 $\rightarrow$ 1.001 |
| 1.475 mm, 25 $\mu$m | 1.007 $\rightarrow$ 1.001 | 1.0009 | 1.031 $\rightarrow$ 1.001 |
| 1.475 mm, 12.5 $\mu$m | 1.004 $\rightarrow$ 1.001 | 0.9998 | 0.594 $\rightarrow$ 1.001 |
| 0.840 mm, 25 $\mu$m | 1.013 $\rightarrow$ 1.002 | 1.0016 | 1.036 $\rightarrow$ 1.002 |

### Shield material and thickness sensitivity

For the 0.9/2.95 mm PTFE coax geometry above, replacing only the copper shield with
aluminium changes attenuation by 5.6--9.8% from 10 to 500 MHz.  This material choice is
therefore comparable to the Tesche--Schelkunoff model difference and cannot be absorbed
as a negligible construction detail.  The calculation applies Schelkunoff surface
impedance independently to the two conductor materials
([Schelkunoff, 1934](https://doi.org/10.1002/j.1538-7305.1934.tb00679.x)).

| Shield construction | 10 MHz attenuation change | 500 MHz attenuation change |
|---|---:|---:|
| Aluminium instead of copper | +5.6% | +9.8% |
| Copper wall $\ge25\ \mu$m | <0.01% | <0.01% |
| Copper wall 12.5 $\mu$m | +16% | <0.1% |

The thickness calculation shows the opposite priority: walls of 25 $\mu$m or more are
unchanged to four decimal places in this band, while a 12.5 $\mu$m foil matters at the
low end.  Finite wall thickness remains useful for correctness and lower frequencies,
but dissimilar shield material is the larger 10--500 MHz sensitivity
([Schelkunoff, 1934](https://doi.org/10.1002/j.1538-7305.1934.tb00679.x)).
