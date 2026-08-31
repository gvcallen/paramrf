# AGENTS.md

**ParamRF** (`pmrf`) is a JAX-native RF modelling framework: frequency-domain circuit
simulation, optimisation, fitting, and inference. Models are Equinox modules — immutable
dataclasses that are also JAX PyTrees. Built on `jax`, `equinox`, and
[`parax`](https://gvcallen.github.io/parax) (the parameter/constraint layer).

## Breaking changes are acceptable

The library is pre-1.0 and moves quickly. Do not add backwards-compatibility shims,
deprecation aliases, or legacy code paths unless explicitly asked. Prefer the clean design.

## Commands

```bash
.venv/bin/python -m pytest        # full suite
.venv/bin/python -c "import pmrf" # import smoke check
```

No linter or formatter is configured. Match the style of surrounding code.

## Source layout

Real code lives in `pmrf/`. `build/` and `dist/` hold stale copies of it — searches will
hit them, but never read or edit them as source.

## Naming

- `Solver` — reserved for classes that actually solve a system (`GlobalMNACircuitSolver`).
- `Formulation` — closed-form physics strategy objects.
- `Abstract` — prefixes an ABC unless it is domain terminology. `Model` is unprefixed; it is closer to a mix-in.

## Documentation

Numpydoc docstrings. Maths is written `$$...$$` and rendered by `sphinx-math-dollar`.
Physics classes carry a `**Mathematical Formulation**` section stating the equations they
implement and a `References` section citing the source paper — match that, it is the main
defence against an unattributed formula drifting. Fields are documented with `#:` comments
directly above them.

## Tests

`tests/` mirrors the package. When touching physics, validate against `scikit-rf` rather
than against recorded ParamRF output, which only locks in current behaviour. Record
tolerances per case; do not loosen a global tolerance to make one case pass.

## Agent skills

### Issue tracker

Issues live as GitHub issues on `gvcallen/paramrf`, managed with the `gh` CLI. See
`docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root, both created lazily. See
`docs/agents/domain.md`.
