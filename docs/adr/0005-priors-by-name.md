# ADR-0005: Priors are attached by name, and a joint prior keeps its parameters

Status: accepted (2026-09)

## Context

A fit's posterior is often the next fit's prior. The driving case: a normalizing
flow is trained on one fit's samples, over 20 to 40 named free parameters, and
must be the prior of the next fit under MCMC and under PolyChord. A flow is one
example; the requirement is any joint distribution over named parameters.

`prf.modules.Probabilistic` was the only way to attach a joint prior. It
replaces its target sub-tree with a `parax.Probabilize` node, which unwraps the
sub-tree and stores one raw value for all of it. That fitted Parax's own style,
where parameters have no names and a sampler moves through a partitioned
pytree. It does not fit ParamRF since ADR-0002, where everything is addressed by
parameter name and solvers see a name-keyed dict:

- The covered parameters lose their names. A prior over the root is one
  parameter named `''`, valued as a whole model.
- A `Fixed` parameter inside the target breaks construction.
- The name resolver, `log_prior` and `resolve` special-case the collapsed node.
- Hypercube samplers reject it, since it is not a `Param` with an `icdf`.
- A joint prior over parameters in different sub-models needs their common
  parent as the target, and absorbs everything below it.

The one thing it does well must be kept: its raw value is the distribution's
whitened space (for a flow, its base `z`), so optimisers and samplers move in
well-conditioned coordinates.

## Decision

### `prf.prior`

`prf.prior(tree, names, distribution, space='declared')` returns a copy of a
tree with a prior attached to the parameters that `names` selects. `names` is a
selector, as for `prf.update`. It is the one way to attach a prior after
construction, whether one-dimensional or joint, and its form follows `prf.tie`.

The distribution's event size decides which:

- **Scalar event**: each selected parameter gets its own prior, as if built with
  `prf.Random`. Its raw space becomes the new prior's whitening, as for
  `prf.Random`, and the prior is truncated to the parameter's existing bounds so
  the bounds still hold.
- **Event size equal to the number of selected parameters**: a **joint prior**
  over them.
- Anything else raises.

### A joint prior keeps its parameters

A joint prior is a wrapper around the unchanged tree, holding the distribution,
the parameter names and the space: `Probabilistic`, rewritten. The wrapper is
transparent to the name resolver, as `Tied` is. So:

- Every parameter keeps its name and stays a parameter. `prf.params`,
  `prf.param_values` and `prf.update` work as before.
- The prior stores names, not paths. With an explicit sequence of names, the
  distribution's vector is in that order; a glob expands to sorted names.
- Several joint priors may be attached, one wrapper each, over disjoint sets of
  parameters. A parameter under two joint priors raises.
- The joint prior **replaces** its parameters' own priors. Multiplying them in
  would count the earlier fit's priors twice, since its posterior already
  includes them.

### Space

`space` states the space the distribution is over: `'raw'`, `'declared'` or
`'physical'`. ParamRF applies the mappings between spaces, with their
change-of-variables terms.

`'raw'` means the parameters' raw space as it was just before `prf.prior` was
called, which is what `prf.param_values(..., space='raw')` returned then. A
distribution over raw space is only meaningful with respect to that raw to
declared mapping, so the mapping is kept when the prior replaces the
parameters' own priors, including any whitening their old priors gave it.

### Attaching a joint prior redefines raw space

For the parameters under a joint prior, raw space becomes the distribution's
whitened space, inferred as for a one-dimensional prior
(`parax.constraints.infer_distribution_constraint`): a flow's base `z`, the
Cholesky-whitened space of a multivariate normal, and so on. Where no whitening
is known, raw is the distribution's own space. Raw space is ParamRF's to define:
it is a latent space, usually whitened to some degree. Each parameter's raw value
is its entry in that vector, so it keeps its name.

The same word, raw, therefore names two spaces for these parameters: the one
`space='raw'` refers to (before attaching), and the one solvers move in (after).

### Bounds

A parameter's bounds still hold under a joint prior, and its prior is exactly
normalised, so a hypercube sampler's evidence is exact. Solvers move in the
whitened space, which reaches the distribution's whole support, and a parameter
cannot hold a value outside its bounds, so the support must already lie inside
them:

- A distribution over raw space cannot leave the bounds: it reaches declared
  space through the parameters' old raw to declared mappings. This is the
  intended use, a flow trained on an earlier fit's raw samples.
- A distribution over declared or physical space is rejected by `prf.prior`
  unless its support lies inside the parameters' bounds. A joint distribution
  cut at the bounds has no exact normaliser, and an unnormalised one would bias
  the evidence.

The check is against the parameters' current bounds. These do not yet record
whether a bound is the model's (validity, such as a positive width) or the
user's (a range, such as `prf.Bounded`), so the check treats every bound as
validity.

### Hypercube samplers

A hypercube sampler maps the unit cube through a joint prior's block as
`u -> Normal(0, 1).icdf(u) -> whitening -> raw -> declared` whenever the
whitening ends in an independent standard-normal base. Parameters outside
joint priors keep their own `icdf`. A joint prior without such a base raises,
saying which structure is needed. This is built in ParamRF's sampler, not as a
distribution's `icdf`: the map is a cube transform, not a multivariate inverse
CDF.

### Fixing and tying

- Fixing a parameter under a joint prior raises. Its fixed value does not fix
  any one raw coordinate, and a hypercube sampler cannot draw from the
  conditional. To fix it, marginalise the distribution first and attach it over
  fewer parameters.
- Tying a parameter under a joint prior raises, whichever is applied first: the
  distribution must control it.
- `prf.resolve` keeps a joint prior, since it is not a tie and does not change
  what the parameters are. `prf.unwrap` drops it, as it drops every prior.

## Rejected options

- **Polishing the collapsing `Probabilistic`** (keeping the names of the
  parameters it absorbs). Names, `Fixed` and hypercube support all fight the
  collapse; the collapse is what has to go.
- **A joint prior as a separate argument** (`prior=` on `log_prior` and
  `sample`). Every consumer would have to pass it on, `prf.log_prior(model)`
  would be silently wrong without it, and a saved model would lose its prior.
- **Solvers moving in the old raw space under a joint prior.** Simpler, but badly
  conditioned for a flow, which is the main reason to use one.
- **Allowing fixed parameters under a joint prior for MCMC only.** The joint
  density at the fixed value is the right conditional up to a constant, but with
  whitened raw coordinates a fixed parameter fixes no coordinate.
- **Calling the pre-attach space `'unconstrained'`.** Inaccurate: it includes
  the whitening from the old priors. The term is also on the avoid list.
- **`Transformed.icdf` in distreqx.** It would not be an inverse CDF, and it would
  put ParamRF's sampler logic in another repository.

## Consequences

- The old `Probabilistic` constructor (`target=`, `static=`, tuple targets) and
  the `parax.Probabilize` node are removed, with no alias, along with the
  special cases for them in the name resolver, `log_prior` and `resolve`.
- ADR-0003's claim that a joint prior over a derived model's base scores
  unchanged still holds, and more simply: the base's names are the prior's names.
- A distribution over raw space is tied to the constraints and priors the
  parameters had when it was trained. Changing them between fits changes what
  it means, and ParamRF does not check this.
- ParamRF no longer uses Parax's subtree-level `Probabilize`. What Parax keeps
  is a separate decision.
