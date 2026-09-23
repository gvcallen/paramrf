# AGENTS.md

**ParamRF** (`pmrf`) is a JAX-native RF modelling framework: frequency-domain circuit
simulation, optimisation, fitting, and inference. Models are Equinox modules — immutable
dataclasses that are also JAX PyTrees. Built on `jax`, `equinox`, and
[`parax`](https://gvcallen.github.io/parax) (the parameter/constraint layer).

## Breaking changes are fine

Pre-1.0. No backwards-compatibility shims, deprecation aliases or legacy code paths
unless asked. Prefer the clean design, within the issue's seam.

## Seams

Each issue names one seam: a file, a set of files, or a folder. If the work needs a
change outside it, stop and ask.

`CONTEXT.md` and ADRs (architecture decision records, `docs/adr/`) are edited only by
the reviewer, in their own commit. If you need a domain term that isn't in the issue,
an ADR or `CONTEXT.md`, that's a design decision: stop and ask.

Start every PR body with `## Cross-seam`, listing each file changed outside the issue's
seam with the ADR or issue line that allows it, or `DECISION NEEDED`. Write `none` if
there are none.

## Commands

```bash
.venv/bin/python -m pytest        # full suite
.venv/bin/python -c "import pmrf" # import smoke check
```

No linter or formatter is configured. Match the style of surrounding code.

## Commits

Do not add yourself as an author. No `Co-Authored-By` trailer, no session link, no
tool attribution.

## Source layout

Real code lives in `pmrf/`. `build/` and `dist/` hold stale copies of it — searches will
hit them, but never read or edit them as source.

## Naming

- `Solver` — reserved for classes that actually solve a system (`GlobalMNACircuitSolver`).
- `Formulation` — closed-form physics strategy objects.
- `Abstract` — prefixes an ABC unless it is domain terminology. `Model` is the exception.

## Documentation

Numpydoc docstrings. Maths is written `$$...$$` and rendered by `sphinx-math-dollar`.
Physics classes carry a `**Mathematical Formulation**` section stating the equations they
implement and a `References` section citing the source paper; match that. Fields are
documented with `#:` comments directly above them.

## Tests

`tests/` mirrors the package. When touching physics, validate against `scikit-rf` rather
than against recorded ParamRF output, which only locks in current behaviour. Record
tolerances per case; do not loosen a global tolerance to make one case pass.

**scikit-rf is guidance, not ground truth.** It makes its own approximations and
modelling choices, and it has been wrong before. When a comparison disagrees, read the
scikit-rf source (not just its docstring) and check both implementations against the
cited paper. If ParamRF is wrong, fix it; if scikit-rf is, or the two deliberately model
different things, say so in the test with the reason. Never widen a tolerance until the
difference is explained.

## Issues and docs

### Issue tracker

Issues live as GitHub issues on `gvcallen/paramrf`, managed with the `gh` CLI. See
`docs/agents/issue-tracker.md`.

### Triage labels

See `docs/agents/triage-labels.md`.

### Domain docs

One `CONTEXT.md` and one `docs/adr/` at the repo root; either may not exist yet. See
`docs/agents/domain.md`.
