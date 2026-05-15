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

In general, ParamRF follows a tiered API in order to isolate the JAX and solver logic. It is useful to understand the module split (and perhaps have a basic grasp on JAX PyTrees and perhaps the `Equinox` library) before continuing. A brief overview on the different modules and some JAX "gotchas" is given below.

### The Module Split
Higher-level methods (fitting, optimization etc.) should be available to the user with minimal imports. However, these routines shuold build on top of "glue" helpers so that power users can pass additional arguments (such as custom evaluators in `pmrf.evaluators`) to customize the algorithm.

* **Core Primitives** (`pmrf.Model`, `pmrf.Frequency`, `pmrf.parameters`). These are the core primitives in the library. They represent the foundational classes and functions that both users and library routines will interact with.
* **Lower-level Routines** (`pmrf.math`, `pmrf.rf`). The functions implement either simple computations or highly-specialized numerical algorithms. They should all operate directly on JAX arrays (i.e. not PyTrees).
* **RF Models** (`pmrf.models`). These are the main, built-in RF models in the library. These have been separated into an organizational structure (e.g. lumped components, surrogates etc.) and new models should align with this structure. These models should only implement the forward pass e.g. taking the model parameters and producing their RF response. Any inverse design/solution should be placed under "algorithms".
* **Algorithms** (`pmrf.optimize`, `pmrf.infer`, and `pmrf.solve`). These are a combination of functional entry points, wrappers around external libraries and (hopefully soon) direct, RF-specific solvers. Refer to the optimization module (specifically `pmrf.optimize.base` and `pmrf.optimize.minimize`) for a good example. This module splits optimization algorithms into three layers, namely the solver class, the base minimization routine, and the higher-level minimization routine. The solver class implements an abstract interface which only operates on JAX arrays and PyTrees and NOT library primitives. This creates a clean separation between the library-specific entry point (the free function that understands library primitives) and the maths (the solver). The base routine then performs any Equinox-specific translations while still remaining agnostic to the specific JAX PyTree, and the higher-level routine translates library primitives to the base layer. This tiered approach makes testing a lot easier, and also helps isolate the RF logic (setting up the objective function) from the JAX logic.
* **Toolkits** (`pmrf.evaluators`, `pmrf.losses`, `pmrf.likelihoods`, `pmrf.discrepancy_models`, `pmrf.noise_models`, `pmrf.covariance_models`). These are the concrete "glue" components that provide users with a toolkit to mix-and-match together in order to solve their problem, mostly for power users.
* **Routers** (`pmrf.fitting`) This is an example of a high-level convenience API specifically for fitting models to measured data. Functions here, like `fit`, `fit_minimize` or `fit_sample`, essentially act as "routers", converting different data formats (e.g. `skrf.Network`) and user specs into code that other modules in the library can understand.

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