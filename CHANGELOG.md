# Changelog

## Unreleased

- Complete Tesche coaxial conductor physics across the low-frequency regime,
  and pass evaluated material properties to pure coaxial formulations.
- Correct microstrip results by enabling Kirschning--Jansen modal dispersion by
  default. Set `dispersion=None` on `MicrostripLine` to retain the quasi-static
  pipeline explicitly.
- Add the Hammerstad--Jensen microstrip formulation and selectable complex
  (ADS-like) and real (QUCS-like) permittivity conventions.

## 0.35.0

Parameter names, values and serialisation (#138). Downstream users pinning an
exact version: every breaking change below ships in **0.35.0**. `prf.flatten`
(#135) is not included in this release.

### Breaking changes (0.35.0)

- **Names (#133):** `named_params()` no longer lists tied targets; a tie's
  target is derived, not stored.
- **Names (#133):** wrapper path parts (`Tied`, `Probabilistic`, `Wrapped`) are
  dropped from parameter names.
- **Names (#133):** `.at` and `.tied` raise on a name collision.
- **Names (#133):** `prf.unfreeze` also unfreezes frozen parameters inside the
  value.
- **Names (#133):** string dict keys that are valid identifiers give dotted
  names (`components.cable.length`) instead of the bracket form.
- **Values (#134):** `prf.replace(p, value=...)` takes the physical, scaled
  value, so `replace(p, value=p.value)` returns the same parameter.
- **Values (#134):** the `Param` constructor rejects `raw_value` together with
  `distribution` or `constraint`.
- **Save format (#136):** prf files carry a header
  (`{"format": "prf", "schema_version": 1, "paramrf_version": ..., "tree": ...}`).
  `prf.load` rejects files without it, so files saved by earlier versions
  cannot be loaded.
- **Save format (#136):** `prf.save` writes every field, defaults included.

### Added

- One name resolver behind `named_params`, `.at`, `.tied` and
  `tree_param_names_to_path`; names resolve through frozen parameters and after
  evaluating a `Touchstone`-backed model (#133).
- `prf.replace` works on `Param`, keeping prior, constraint, fixed state, scale,
  name and metadata (#134).
- `Module.values`, `with_values`, `with_free` and `with_fixed` (#134).
- `pmrf.serialization.SCHEMA_VERSION` and `FORMAT` constants, and a clearer
  `ImportError` when a saved class cannot be imported (#136).

### Fixed

- `pmrf.__version__` is now set; it was looked up under the wrong
  distribution name (#136).
- The `Model.build` deprecation warning is emitted once per class, and now
  points to plain functions for composites with no parameters of their own
  (#137).

### Documentation

- `Module` and `Substrate` explain that passing the same instance to two
  sibling fields gives independent parameters, and how to share one (#137).
