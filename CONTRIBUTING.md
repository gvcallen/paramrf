# Contributing to ParamRF

First off, thanks for considering contributing! Bug fixes, new models and new solvers are always appreciated.

ParamRF builds on top of JAX and Equinox's functional style. If you are coming from standard Python object-oriented programming, there are a few JAX-specific paradigms to keep in mind, which are outlined below.

## Setting up for Local Development

1. **Fork and Clone:** Fork the repository on GitHub and clone it locally.
2. **Virtual Environment:** Set up a virtual environment using Python 3.11+.
3. **Install Dependencies:** Install the package in editable mode along with the test and documentation dependencies:

    pip install -e .[tests,docs]

4. **External Inference Dependencies (Optional):** If you plan on working on the Bayesian inference module (`pmrf.infer`), you may need to install our custom `distreqx` fork and PolyChord. Note that PolyChord requires C++ and Fortran compilers (like `mpicxx` and `mpifort`) to be installed on your system:

    ```bash
    pip install git+https://github.com/gvcallen/distreqx.git
    ```

    ```bash
    pip install git+https://github.com/PolyChord/PolyChordLite.git
    ```

## Building docs/running tests

We use `pytest` for all unit testing. Simply run:

    pytest

When writing new tests, especially for fitting and inference routines, try to use synthetic, in-memory S-parameter data (identity fits) rather than committing `.s2p` files to the repository. This keeps the test suite fast and the repository size small.

For docs, we use sphinx. Simply run:

    make html

## Architecture and Design Philosophy

In general, ParamRF follows a tiered API. Higher-level methods should always be available to perform a task with minimal imports. However, these should rely on mid-tier functions in the library that use "glue" helper classes which power-users can also use.

### The Module Split
* **Core Primitives** (`pmrf.core`, `pmrf.math`, `pmrf.rf`). These are the lower-level primitives used throughout the library. The functions here should implement either very simple computations, or highly-specialized numerical algorithms. These should all operate directly on JAX arrays (i.e. not PyTrees). The core base classes define interfaces for re-useable "glue" components to be implemented in the rest of the library (explained further below).
* **RF Models** (`pmrf.models`). These are the main, built-in RF models in the library. These have been separated into an organizational structure (e.g. lumped components, surrogates etc.) and new models should align with this structure. These models should only implement the forward pass e.g. parameters to RF response. Any inverse design/solution should be placed under "algorithms" (discussed below). See the folder structure for a better idea of the model layout.
* **Algorithms** (`pmrf.optimize`, `pmrf.infer`, and `pmrf.solve`). These are a combination of functional entry points, wrappers around external libraries and (hopefully soon) direct, RF-specific solvers. Free functions should be used here as entry points for accomplishing a specific task. These should be independent of the underlying algorithm (e.g. "non-linear minimization" or "bayesian sampling") and should be named using verbs (e.g. "minimize" or "sample"). These functions should accept callable PyTrees, JAX arrays, library primitives, and a solver instance as their main arguments, and return a high-level "results" object interpretable by the user. For a general example, see `pmrf.optimize.minimize` or `pmrf.infer.sample`, which demonstrate this for external solver functions (`optimistix` and `inferix`) and built-in solvers/wrappers (`pmrf.optimize.ScipyMinimize` or `pmrf.infer.PolyChord`). Any new solvers (custom or wrapper) should derive from `eqx.Module` and contain their settings as dataclass fields so that they can be instantiated independent of the specific problem. Note that the solvers themselves should only accept and return JAX arrays and PyTrees, and NOT library primitives! This creates a clean separation between the library-specific entry point (the free function that understands library primitives) and the maths (the solver).
* **Toolkits** (`pmrf.evaluators`, `pmrf.losses`, `pmrf.likelihoods`, `pmrf.discrepancy_models`, `pmrf.noise_models`, `pmrf.covariance_models`). These are the concrete glue components that provide users with a toolkit to mix-and-match together in order to solve their problem. Both high-level and power users may interact with any of these.
* **Fitting** (`pmrf.fitting`) This is an example of a high-level convenience API specifically for fitting models to measured data. Functions here, like `fit`, `fit_minimize` or `fit_sample`, essentially act as "routers", converting different data formats (e.g. `skrf.Network`) and user specs into code that other modules in the library can understand.

If you are adding a completely new algorithm, a few things need to be considered. For example, lets consider the case of adding vector fitting to the library (PRs welcome!).
- We first need to consider what general, lower-level task the algorithm solves. Since this is a specialized solver, it would go under `pmrf.solve`.
- The lower-level task here is "functional approximation". We would therefore would create a free function to match this, perhaps named "approximate" under `pmrf.solve.approximate`. This would likely accept the data, frequency and solver as input, and return the approximate model and solver statistics as output. Note that since vector fitting belongs to a family of approximation known as rational approximation, it might make sense to create a helper function `approximate_rational` that understands the raw format returned by the solver and creates the appropriate model (e.g. `pmrf.models.surrogates.rational.PoleResidue`), and routes to that solver from `pmrf.solve.approximate`.
- Finally, implement the algorithm in a class e.g. `pmrf.solve.VectorFitter`. Unless the algorithm could be written to match some specific high-level contract, this would probably contain a single `__call__` method, and return a results structure with the final solution, as well as any solver-specific statistics (e.g. convergence).
- (optional) In this case, you may want to implement a higher level function e.g. (`pmrf.fitting.fit_rational`) for easier discovery, since rational approximation's main purpose is fitting measured data.

Note that `pmrf.Problem` may be a helpful primitive when implementing general callable algorithms.

### JAX and Equinox "Gotchas"
* **Immutability:** `Model` classes inherit from `equinox.Module`. They are therefore immutable and classes are structured around this.
* **Pure Functions:** JAX requires functions to be pure (no side effects) to be compiled with `@jax.jit`. Make sure to use the `jax.lax` module where possible.
* **Shapes and Types:** The library uses (nfreq, nports, nports) throughout RF functions, but swaps to nfreq on the last axis for statistical functions. This enables easy batching using `jax.vmap`.

## Submitting a Pull Request

1. Create a new branch from `main` (e.g., `git checkout -b feature/new-solver`).
2. Write tests for your new feature or bug fix.
3. Ensure the entire test suite passes (`pytest`).
4. Push your branch and open a Pull Request on GitHub. 

Provide a clear description of what the PR does. If it fixes an open issue, link to it in the description!