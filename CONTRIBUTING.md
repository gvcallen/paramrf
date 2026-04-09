# Contributing to ParamRF

First off, thanks for considering contributing! Bug fixes, new models and new solvers are always appreciated.

ParamRF follows a strict module API split, and also builds on top of JAX and Equinox's functional style. If you are coming from standard Python object-oriented programming, there are a few JAX-specific paradigms to keep in mind, which are outlined below.

## Setting up for Local Development

1. **Fork and Clone:** Fork the repository on GitHub and clone it locally.
2. **Virtual Environment:** Set up a virtual environment using Python 3.11+.
3. **Install Dependencies:** Install the package in editable mode along with the test and documentation dependencies:

    pip install -e .[tests,docs]

4. **External Inference Dependencies (Optional):** If you plan on working on the Bayesian inference modules (`pmrf.infer`), you may need to install our custom `distreqx` fork and PolyChord. Note that PolyChord requires C++ and Fortran compilers (like `mpicxx` and `mpifort`) to be installed on your system:

    pip install git+https://github.com/gvcallen/distreqx.git
    pip install git+https://github.com/PolyChord/PolyChordLite.git

## Running the Tests

We use `pytest` for all unit testing. Simply run:

    pytest

When writing new tests, especially for fitting and inference routines, try to use synthetic, in-memory S-parameter data (identity fits) rather than committing `.s2p` files to the repository. This keeps the test suite fast and the repository size small.

## Architecture and Design Philosophy

In general, ParamRF follows a tiered API. Higher-level methods should always be available to perform a task with minimal imports. However, these should rely on mid-tier functions in the library that use "glue" helper classes which power-users can also use.

### The Module Split
* **High-level**: (*`pmrf.fitting`*) This is the user-facing data fitting API. Functions here (like `fit`, `fit_minimize` or `fit_sample`) essentially act as "routers", converting different data formats and user specs into code that the mid-tier methods understand.
* **Mid-tier** (*`pmrf.optimize`, `pmrf.infer` and `pmrf.solve` for direct solvers*). These are pure "routine" modules or backend translation layers. They should accept mostly callable PyTrees and JAX arrays. They should never interact with higher-level formats or convenience methods, like string alias, built-in losses and likelihoods, external data libraries like scikit-rf, etc.
* **Low-level** (*`pmrf.math`, `pmrf.rf`, `pmrf.core`). These are raw, input-output functions and algorithms that are used throughout the library, or the core abstract base classes. They either implement relatively simply algorithms working on raw JAX arrays, or define interfaces for the rest of library.
* **Built-ins** (*`pmrf.models`, `pmrf.losses`, `pmrf.likelihoods`, `pmrf.evaluators`, `pmrf.discrepancy_models`, `pmrf.noise_models`). These are "jack of all trades" bundled libraries that the provide the user with a toolkit to mix-in-match and create the algorithm they desire. Both high-level and power users may interact with any of these.

If you are adding a new algorithm (optimization, inference or direct), implement the math in the appropriate solver module and update the `pmrf.fitting` router to support it.

### JAX and Equinox "Gotchas"
* **Immutability:** `Model` classes inherit from `equinox.Module`. They are therefore immutable and classes are structured around this.
* **Pure Functions:** JAX requires functions to be pure (no side effects) to be compiled with `@jax.jit`. Make sure to use the `jax.lax` module where possible.
* **Shapes and Types:** The library uses (nfreq, nports, nports) throughout RF functions, but swaps to nfreq on the last axis for statistical functions. This enables batching using `jax.vmap`.

## Submitting a Pull Request

1. Create a new branch from `main` (e.g., `git checkout -b feature/new-solver`).
2. Write tests for your new feature or bug fix.
3. Ensure the entire test suite passes (`pytest`).
4. Push your branch and open a Pull Request on GitHub. 

Provide a clear description of what the PR does. If it fixes an open issue, link to it in the description!