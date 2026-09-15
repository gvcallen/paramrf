# Changelog

## 0.35.0

Parameter names, values and serialisation (#138), and the parameter API of
ADR-0002 (#146). Downstream users pinning an exact version: every breaking
change below ships in **0.35.0**.

Name-keyed operations are now free functions rather than methods, values are
read and written in the units they were declared in, and changing a value no
longer recompiles a model's RF methods.

### Breaking changes (0.35.0)

- **Parameter API (#146):** name-keyed operations are free functions, so they
  also work on a collection such as `(model, noise_model)`. Removed:
  `Module.named_params`, `values`, `with_values`, `with_free`, `with_fixed`,
  `map` and `at`; `Module.tied` and `Model.tied`; `Param.as_fixed`, `as_free`
  and `at`; and the public `Param.wrap`. Use `prf.params`, `prf.param_values`,
  `prf.update` and `prf.tie`. There are no aliases or deprecation shims.
- **Parameter API (#146):** `Module` has no public methods. `Model` keeps its RF
  methods, which are now fixed by a checked-in allowlist.
- **Values (#146):** `Param.value`, `prf.param_values` and `prf.update` use the
  **declared** value, the number as written in the units the parameter's scale
  declares, so `prf.update(m, {'C': 3.0})` on a pF parameter sets 3 pF, not 3 F.
  This reverses the physical-value behaviour `prf.replace(p, value=...)` had
  earlier in this release.
- **Values (#146):** a scale on a value overrides the field's scale instead of
  multiplying with it, so a `Param` with `scale=1e-9` passed into a pF field is
  nF. `Param.scale` is `None` when the parameter declares none.
- **Values (#146):** `Param.unscaled_value` is removed; the three spaces are
  `Param.value` (declared), `physical_value` and `raw_value`.
- **Values (#146):** the `Param` field holding the Parax variable is renamed
  from `raw_value` to `variable`, and the bijector properties are now
  `raw_to_declared_bijector` and `declared_to_physical_bijector`.
- **Values (#146):** a model's `repr` shows declared values, not scaled ones.
- **Inference (#146):** the minimiser and samplers run on raw-space values, over
  a name-keyed dict rather than a flat vector. `pmrf.utils.optix`'s lens is
  internal; nothing public returns a `Lens`.
- **Inference (#146):** `prf.log_prior` is a density in the space asked for, so
  a raw-space prior carries the constraint's Jacobian and scale terms. Saved
  results that stored physical values need re-reading in declared space.
- **Names (#133):** parameter names no longer include tied targets; a tie's
  target is derived, not stored.
- **Names (#133):** wrapper path parts (`Tied`, `Probabilistic`, `Wrapped`) are
  dropped from parameter names.
- **Names (#133):** a name collision raises.
- **Names (#133):** `prf.unfreeze` also unfreezes frozen parameters inside the
  value.
- **Names (#133):** string dict keys that are valid identifiers give dotted
  names (`components.cable.length`) instead of the bracket form.
- **Names (#146):** a named model held in a dict drops its key, and top-level
  dict keys do not add a namespace layer.
- **Values (#134):** the `Param` constructor rejects a raw value together with
  `distribution` or `constraint`.
- **Save format (#136):** prf files carry a header
  (`{"format": "prf", "schema_version": 1, "paramrf_version": ..., "tree": ...}`).
  `prf.load` rejects files without it, or with a `schema_version`
  other than 1, so files saved by earlier versions
  cannot be loaded.
- **Save format (#136):** `prf.save` writes every field, defaults included.

### Added

- `prf.params` and `prf.param_values` read a model's parameters by name, and
  `prf.log_prior` scores them, each over a selector and one of the three value
  spaces (#146).
- `prf.update`, one verb for changing a model: values by name, a value or fixed
  state over a selector, a new sub-model, or a function of the old part (#146).
- `prf.tie` as a free function (#146).
- One name resolver behind every name-based operation; names resolve through
  frozen parameters and after evaluating a `Touchstone`-backed model (#133).
- `prf.replace` works on `Param`, keeping prior, constraint, fixed state, scale,
  name and metadata (#134).
- `pmrf.serialization.SCHEMA_VERSION` and `FORMAT` constants, and a clearer
  `ImportError` when a saved class cannot be imported (#136).

### Fixed

- Changing a parameter's value no longer recompiles a model's RF methods. The
  value forms of `prf.update` keep the tree structure and every leaf's dtype,
  shape and weak type (#130, #146).
- Compiling `model.s(freq)` for a Touchstone-backed model no longer takes
  minutes: interpolation is vectorised over port pairs rather than looped. At
  50 ports with linear interpolation, the first call drops from 70 s to 0.4 s;
  100 ports no longer times out. S-parameter data stays static, which keeps
  per-call evaluation fast (#130, #152).
- `pmrf.__version__` is now set; it was looked up under the wrong
  distribution name (#136).
- The `Model.build` deprecation warning is emitted once per class, and now
  points to plain functions for composites with no parameters of their own
  (#137).
- `pmrf.__all__` no longer lists `Topology` and `Initvar`, which do not exist
  (#146).

### Documentation

- New "Working with parameter names" page in core concepts: how names are
  formed, the three value spaces, reading, changing, and what recompiles. It
  replaces the "Parameter naming and model manipulation" example page (#146).
- `Module` and `Substrate` explain that passing the same instance to two
  sibling fields gives independent parameters, and how to share one (#137).

## Unreleased

- Complete Tesche coaxial conductor physics across the low-frequency regime,
  and pass evaluated material properties to pure coaxial formulations.
- Correct microstrip results by enabling Kirschning--Jansen modal dispersion by
  default. Set `dispersion=None` on `MicrostripLine` to retain the quasi-static
  pipeline explicitly.
- Add the Hammerstad--Jensen microstrip formulation and selectable complex
  (ADS-like) and real (QUCS-like) permittivity conventions.
