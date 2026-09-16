# ADR-0002: Parameter API: free functions, one update verb, three value spaces

Status: accepted (2026-09)

## Context

0.35.0 (#138) made parameter names the key for saved results, ties, free sets
and external tooling, and added name-keyed operations as `Module` methods:
`values`, `with_values`, `with_free` and `with_fixed`, next to the existing
`named_params`, `at`, `map` and `tied`. An older ParamRF had `with_*` methods
and deprecated them because they inflated the API and cut against a move to a
functional style. There was no rule for which operations get a method, so the
surface grew case by case.

Four problems forced a decision:

- **Joint trees.** Fitting and inference increasingly work on collections such
  as `(model, noise_model)`. A method on `Module` cannot reach them, so every
  method-only operation needs a function as well, and the API grows two
  spellings.
- **Scale.** Construction takes the declared value (`prf.param(scale=1e-12)`
  then `RC(1.0, 2.0)` stores 2 pF), but `Param.value`, `values()`,
  `with_values()` and `prf.replace(p, value=...)` use the physical, scaled
  value. `with_values({'C': 3.0})` on that model silently sets 3 F. A `Param`
  with its own scale passed into a scaled field multiplies the two scales
  (`1e-9 × 1e-12`).
- **Recompilation (#130).** RF methods are wrapped in `eqx.filter_jit`, so any
  change to a model's static structure recompiles. Rebuilding a parameter with
  `model.at(name).set(prf.Unconstrained(v))`, as the docs showed, changes its
  scale and constraint and recompiles; for a large circuit that is a minute per
  change. `_replace_raw_value` also flips `weak_type` on the value leaf, so
  `with_values` and `prf.replace` recompile once even though the treedef is
  unchanged.
- **Flattening (#135, #142).** A proposed `prf.flatten` returned a
  `FlatParams` object over a 1-D vector. Flattening that prominent is unusual
  in the JAX ecosystem.

Precedent from other JAX libraries: Equinox's `Module` has no public methods;
Flax NNX deprecated `Module.iter_modules` and `iter_children` in favour of free
functions; Penzai keeps a single `.select()` gateway; NumPyro's
`initialize_model` exposes a potential function over a **name-keyed dict** of
unconstrained values and ravels only inside samplers. scikit-rf, which most
ParamRF users know, puts RF operations on `Network` as methods.

## Decisions

### 1. A method belongs to the class that gives it meaning

If an operation would make equal sense on a plain collection of models and
parameters, such as a `(model, noise_model)` tuple, it is a free function and
never a method. There are no exceptions.

- **`Model`** keeps RF methods: `s`, `a`, `z`, `y`, `mna`, `nports`,
  `port_tuples`, `primary_matrix`, `build`, `expand`, `cascaded`, `flipped`,
  `renumbered`, `terminated`, `**`, `@`, `to_skrf` and `export_touchstone`.
  `tied` leaves; tying is parameterisation, not circuit algebra.
- **`Module`** has no public methods (decision 2).
- **`Param`** keeps read-only properties that describe it (`value`, `bounds`,
  `distribution`, ...). `as_fixed`, `as_free` and `at` are removed; `wrap`
  becomes private.

A test asserts that `Module` has no public methods and that `Model`'s public
methods equal a checked-in allowlist, so adding one is a reviewed decision. The
methods `__init_subclass__` generates (`s_db`, `s_mn_mag`, ...) and the plotting
names `__getattr__` serves are RF and allowed; the test skips them by pattern
and says so. Replacing them is out of scope.

This matches what scikit-rf users expect, where RF operations are methods, and
what JAX users expect, where tree operations are functions.

### 2. `Module` is a contract, not a toolbox

Without methods, `Module` still does work nothing else does. Its `name` is what
makes a named module collapse the path to its left into a namespace, which name
resolution depends on. `pmrf.modules.validate` rejects raw float JAX arrays in
its fields, which are ambiguous between free and fixed parameters. It carries
`metadata`, gives the unwrapped `repr`, and is the base of `Model`, losses,
likelihoods, kernels, materials and evaluators. Subclassing it says the object
takes part in naming and validation. Its docstring says so.

### 3. One verb for changing a model: `prf.update`

`prf.update` returns a copy of a model in which the parts a selector picks are
replaced. Exactly one form says what with:

```python
prf.update(model, {'L1.L': 3.0, 'C1.C': 2.0})      # parameter values by name
prf.update(model, 'L1.L', value=3.0)               # parameter fields on a selection
prf.update(model, 'cable.*', fixed=True)           # fixed state
prf.update(model, 'cascade[1]', Short())           # a new sub-model or node
prf.update(model, 'load.*', fn=lambda p: ...)      # a function of the old part
prf.update(model, v, space='raw')                  # write-back from an optimiser or sampler
prf.update(param, value=3.0)                       # selector omitted: the root itself
```

It replaces `with_values`, `with_free`, `with_fixed`, `Module.map`,
`Module.at`, `Param.as_fixed`, `Param.as_free` and `Param.wrap`.

- **Selectors** are parameter names, `fnmatch` globs over them, sequences of
  names, or callables, resolved by the #133 resolver.
- **Two tiers, set by the form.** The mapping, `value=` and `fixed=` forms go
  through each parameter's constructor: they validate bounds and keep the prior,
  constraint, scale, name and metadata. The node and `fn=` forms are
  structural: they bypass converters and validation, and the docstring says so.
- **The mapping form** is recognised only as the second positional argument
  with every key a string. Its values may be arrays or `Param` objects. Any
  other second argument is a selector, and a form mismatch raises an error that
  lists the forms.
- **`fixed=`** is additive. `fixed=False` frees a parameter even if it was
  created fixed, and parameters the selector does not match are untouched.
  "Only these free" is `update(update(m, '*', fixed=True), names, fixed=False)`.
- **Value forms keep the jit cache key.** The mapping, `value=` and `space=`
  forms never change the treedef, or any leaf's dtype, shape or `weak_type`
  (decision 10). Changing `fixed=`, or any structural form, changes the model's
  structure, and recompiling is expected.
- **Not an optimiser step.** In fitting, "updates" also means Optax gradient
  steps; the docstring says `update` is neither.

`prf.replace` stays as the plain `dataclasses.replace`: fields of one object,
unvalidated, able to break a type. Its docstring points to `update` for
anything name-based.

`prf.tie(model, target, source, fn=identity)` stays a separate verb. A tie is
not a replacement: its target is recomputed from its source every time the
model is unwrapped.

### 4. Reading: `prf.params` and `prf.param_values`

```python
prf.params(model, where='*', *, free_only=False)                          # dict[str, Param]
prf.param_values(model, where='*', *, free_only=False, space='declared')  # dict[str, Array]
prf.log_prior(model, *, space='declared')                                 # scalar
```

A user thinks "the parameters of my model", so `params` returns `Param`
objects, whose `repr` shows value, bounds and prior. `param_values` is what
`update` accepts and what optimisers use; a bare `values` was rejected as too
vague at top level, where it could mean S-parameter data. `named_params`'
`full_params` and `namespace_separator` are dropped.

The documented identity, which is also the regression test of decision 10:

```python
prf.update(m, prf.param_values(m, space=s), space=s)   # same structure, same cache key
```

All functions are defined in `pmrf.parameters` and re-exported at top level.

### 5. Three value spaces: `raw`, `declared`, `physical`

| Space | Meaning | Example (2 pF, `scale=1e-12`) |
|---|---|---|
| `raw` | the latent, unbounded array an optimiser or sampler moves through | constraint bijector inverse of 2.0 |
| `declared` | the number as written, in the units the parameter's scale declares | `2.0` |
| `physical` | the scaled, SI value | `2e-12` |

`declared` is the default everywhere, and every user-facing surface agrees with
construction: `Param(value=...)`, bounds, priors, `repr`, `Param.value`,
`param_values` and `update`. Distributions and bounds are authored in declared
space.

`Param` gets one property per space: `value` (declared), `physical_value` and
`raw_value` (the latent array). `unscaled_value` is removed. The field
currently named `raw_value`, which holds the Parax variable, is renamed
`variable`. The bijector properties become `raw_to_declared_bijector` and
`declared_to_physical_bijector`; "constrained" and "unscaled" are no longer
used as space names.

`log_prior` supports all three spaces: declared is the density as written,
physical adds −log|scale| per parameter, and raw adds the constraint bijector's
log|det J| as well. Each docstring states its measure.

**Why `raw`, not `unconstrained`.** ParamRF's users are mostly RF engineers,
who read "unconstrained" as "a parameter without bounds", which is what
`prf.Unconstrained` creates. The code already used "raw" for this reason.
**Why not `unit`.** In fitting, "unit space" is the unit hypercube nested
samplers work in, which ParamRF's hypercube sampler maps to.
**Why not `unscaled`, `normalized`, `per_unit` or `engineering`.** `declared`
says what the value is rather than what has not happened to it; `normalized`
means z = Z/Z₀ in RF and [0, 1] in optimisation; `per_unit` is power-systems
jargon; "engineering units" means the scaled value in instrumentation.

### 6. Scale is the units a value is written in

A field's scale is a default unit. An explicit scale on the value overrides it;
the two are never multiplied. The factories and `as_param` take `scale=None`,
meaning "inherit the field's", so `prf.Unconstrained(2.0)` passed into a pF
field is 2 pF, and `prf.Unconstrained(2.0, scale=1e-9)` is 2 nF.

This reverses 0.35.0's `prf.replace(p, value=...)`, which took the physical
value; `update(p, value=...)` takes the declared one.

### 7. Fixed and frozen are different things

- **Fixed** is a parameter state. The parameter is still a `Param`, still named
  and still carries its prior; it is excluded from optimisation.
  `update(..., fixed=)` toggles it.
- **Frozen** (`prf.freeze`, `prf.unfreeze`) makes a subtree opaque. It is for
  constant data, such as `field(converter=prf.freeze)`, and for hiding whole
  subtrees.

The name-based operations act on fixed, never on frozen.

### 8. No public flattening

Parameter values cross ParamRF's boundary as name-keyed dicts, which are
pytrees that Optax, Optimistix, BlackJAX and `jax.grad` take directly.
Raveling to a 1-D vector stays inside the adapters that need it, such as the
SciPy solver (`pmrf/optimize/solvers/scipy.py`) and non-JAX samplers.
`prf.flatten` and `FlatParams` (#142) are not added. A public ravel helper, with
one name per array element, can be added when a non-JAX consumer needs it; it
would sit on top of `param_values`, not replace it.

The substance of #142 is kept: the raw-space log prior with the Jacobian and
scale terms, name alignment, and a single implementation behind the minimiser
and samplers.

### 9. Public names describe the domain, not the data structure

Public functions have no `tree_` prefix; most users do not know what a pytree
is and should not need to. `tree_*` names are for private helpers. Docs say "a
model, or any collection of models and parameters", and reserve "pytree" for
an advanced page.

### 10. Recompilation is prevented by keeping structure, not by changing the cache

The automatic `eqx.filter_jit` on RF methods and its cache key stay as they
are. Dropping the automatic jit would make `model.s(freq)` slow for the users
decision 1 keeps methods for; removing `name` and `metadata` from the treedef
would complicate name resolution for a rare case.

Instead, the structure-preserving forms of `update` guarantee an unchanged
cache key, and a test enforces it for every such form: treedef, dtype, shape and
`weak_type` of every leaf. The docs say plainly that changing a parameter's
value never recompiles and rebuilding it does.

## Consequences

- **Breaking, in one minor release.** Removed: `Module.named_params`, `values`,
  `with_values`, `with_free`, `with_fixed`, `map`, `at`, `tied`; `Model.tied`;
  `Param.as_fixed`, `as_free`, `at`, `unscaled_value`; public `Param.wrap`.
  Renamed: the `Param.raw_value` field to `variable`, and the bijector
  properties. Changed: `Param.value` returns the declared value; scales
  override instead of multiplying. No aliases or deprecation shims.
- `pmrf.utils.optix`'s lens becomes internal or is deleted; nothing public
  returns a `Lens`.
- Joint-tree names keep #133's rules. A named model inside a dict drops its key,
  and a collision raises; top-level dict keys do not become a namespace layer.
- Saved results keyed by name keep working; results that stored physical
  values need re-reading in declared space.
- Deferred to their own issues: reducing the top-level and submodule surface
  (including `__all__` listing `Topology` and `Initvar`, which do not exist);
  compile duration for large S-parameter blocks (#130, the per-port
  interpolation loop); the generated `s_db`-style methods; flat vectors for
  non-JAX tools.
