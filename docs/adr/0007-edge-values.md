# ADR-0007: Values at the edge of a domain: zero impedance and parameters on bounds

Status: proposed (2026-09)

## Context

#215 reported zero inductance breaking the default `GlobalMNACircuitSolver`, and
parameters that start exactly on a bound being unable to move. Both are values
at the edge of a domain, and both failed quietly:

- `Inductor.y()` is `1/(jωL)`, which is not finite at L = 0. `Model.mna()`
  stamps from `y()`, so the MNA solve returns NaN.
- With a finite stamp (`PiSectionCLC` goes through ABCD), the MNA solver still
  returns Y, and two ports joined by a short have no finite Y. The solver
  regularises it to about 1e12, and `y2s` of that Y has a condition number
  around 1e14: |S21|² came out 0.7% wrong and dL wrong by ten orders of
  magnitude, without a warning. Removing the regularisation from the auxiliary
  rows made it worse: the singular block went to a least-squares solve, and
  the short came out as an open.
- `BoxSectionCLCC` substitutes eps for L = 0, which fixes the value and zeroes
  the gradient.
- Parax documents `Interval` and `GreaterThan` as exclusive, but validates them
  as inclusive, and maps them onto open raw spaces. So `Bounded(0, 10, value=0)`
  is accepted, has raw value −∞, and cannot move. `minimize` rejects it.
  `Random(Uniform(0, 10), value=0)` is silently moved to 2.2e-15 by a clip in
  parax. `minimize` gives even bounded solvers raw space with infinite bounds,
  so no solver ever sees the box the user wrote.

Engineers expect both to work: an inductor fitted down to 0, and a parameter
started on, or converging to, the bound they gave it.

## Decision

### Circuit solvers return S at the probe reference

A circuit solver that assembles a nodal system loads each external port with
the probe reference impedance $Z_r$ (ADR-0006) and drives it through $Z_r$. With
$H$ the port-voltage response to the drive, it returns power-wave S directly:

$$H = (Y + Z_r^{-1})^{-1} Z_r^{-1}, \qquad
S = F\,\big(2\,\mathrm{Re}(Z_r)\,Z_r^{-1} H - Z_r^{*} Z_r^{-1}\big)\,F^{-1}, \qquad
F = \mathrm{diag}\big(1 / (2\sqrt{\mathrm{Re}\,Z_r})\big)$$

where $Y + Z_r^{-1}$ is eliminated inside the full MNA system, never formed.
Loaded ports keep the system non-singular when ports are shorted together, so
the result is exact there. $Z_r$ may be complex and per-port. For real $Z_r$
this reduces to $S = 2\sqrt{G}\,(Y+G)^{-1}\sqrt{G} - I$; that short form is
wrong for complex $Z_r$.

- **One implementation.** `pmrf.rf.mna2s(stamp, z0)` holds the formula. The
  MNA solver uses it, and so does `Model.s()` for a model whose primary domain
  is MNA.
- **The solve depends on the probe.** `Circuit.s(z0=...)` passes its probe into
  the solve rather than renormalising a probe-free Y afterwards.
- **Regularisation is physical.** GMIN is a conductance from every node to
  ground. Each auxiliary branch gets a +eps Ω series resistance, which resolves
  loops of zero-impedance branches. The system is therefore never singular,
  and the default linear solver is LU (`well_posed=True`): a singular system is
  a bug and shows up as NaN, not as a least-squares answer.
- **Y stays honest.** `Circuit.y()` is `s2y` of S at the native reference, and
  is non-finite for shorted ports. `Circuit.mna()` is `s2mna` of S at the hub
  reference, which stays finite, so nested circuits are exact.
- **Series elements that can short stamp as branches.** An element whose
  impedance can be zero (`Inductor`, `InductorQ`, the inductor in
  `BoxSectionCLCC`) stamps with an auxiliary branch current,
  $V_1 - V_2 - Z I = 0$, not from `y()`. Its `y()` is undefined at zero and may
  be non-finite there. No model substitutes eps for a zero value.
- **The nodal solver stays Y-only.** `GlobalNodalCircuitSolver` cannot
  represent a zero impedance and documents it.

### Bounds are open or closed, and solvers see the box

- **Each bound is open or closed**, declared by its Parax constraint.
  `Positive` and `Negative` are open; `NonNegative` and `NonPositive` are
  closed; `Interval`, `GreaterThan` and `LessThan` take `closed` and default to
  closed. `pmrf.Bounded` and a `Uniform` support are closed. Where two
  constraints meet at one bound, open wins. Validation respects closedness, so
  a value on an open bound is rejected at construction.
- **A closed bound is part of the model's domain.** A model evaluates, with a
  finite value and gradient, at every closed bound of its parameters and at
  zero where zero is valid. A value it cannot evaluate gets an open validity
  constraint instead: a line width is `Positive()`, a quality factor is
  `Positive()`. Lumped L, C and R stay unconstrained, because fitted
  equivalent circuits use negative values.
- **Box space** is a fourth space: the box a bounded minimiser searches. It is
  built from a parameter's bounds, not its prior: the unit box when both bounds
  are finite, declared space otherwise. A closed edge of the box is the bound
  itself; an open edge is inset by δ = 1e-6.
- **Minimisers say whether they honour bounds**, through the minimiser
  interface. `SolverView` hands box space to one that does and raw space to one
  that does not. Nothing in `SolverView` knows about a specific backend.
- **A start on a closed bound is nudged for raw space.** For raw-space
  consumers (unbounded minimisers and samplers), `SolverView` moves a start on a
  closed bound δ = 1e-6 inward in box space before mapping it to raw. The nudge
  is silent: it is the kind of difference engineers expect between optimisers.
- **Raw space is unchanged.** It stays whitened by the prior (ADR-0005).

## Rejected options

- **Keep Y from the MNA solver and detect ill-conditioning.** Wrong answers get
  louder but stay wrong.
- **Substitute eps for zero in models.** It is what `BoxSectionCLCC` did, and
  it zeroes the gradient at exactly the value being fitted.
- **A tanh map for `Interval`.** Any map ending in `lower + width·u` has an
  absolute precision of about eps·width. tanh only moves the exact point from
  the lower bound to the midpoint, and loses it at a lower bound of 0, the
  common case.
- **Reject every start on a bound.** Correct, but engineers write bounds they
  mean to include.
- **Nudge by a fixed raw value.** A raw value of −14 is 1e-6 of a sigmoid's
  width, but 1e-43 of a prior-whitened parameter's slope. The nudge has to be
  in box space to mean the same thing for every parameter.
- **Track whether a bound came from a range or a validity constraint.**
  Closedness is a property of the constraint; recording it there needs no
  bookkeeping when constraints are intersected.
- **Box space whitened by the prior**, as Parax does for `Random`. A prior is a
  belief, not the shape of the search; a `Normal` prior made the box edges ±∞,
  kept finite only by a clip.

## Consequences

- Circuits with shorted ports, and every component at L, C or R = 0, give the
  same S and gradient under MNA and scattering solvers.
- `value=0` on a `Positive` field now raises at construction.
- A `Random` parameter on a bound has raw value ±∞ once Parax drops its clip,
  and is nudged like any other.
- Fits run through a minimiser that honours bounds now evaluate the objective
  exactly on closed bounds. A test sweep evaluates every component at its
  closed bounds and at zero so this is safe.
- A fit that started on a bound gives slightly different results than before.
- `CONTEXT.md` needs **box** under *Space*, and **open** and **closed** bounds.
