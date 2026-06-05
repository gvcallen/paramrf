# Contributing to ParamRF

First off, thanks for considering contributing! Bug fixes, new models and new solvers are always appreciated. Also, if you would just like to open a discussion for a feature request or report a bug, feel free to open an issue on GitHub.

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
Higher-level methods (fitting, optimization etc.) should be available to the user with minimal imports. However, these routines should build on top of "glue" helpers so that power users can pass additional arguments (such as custom evaluators in `pmrf.evaluators`) to customize the algorithm.

* **Core Primitives** (`pmrf.Model`, `pmrf.Frequency`, `pmrf.Param`). These are the library's core primitives. They represent the foundational classes and functions that both users and library routines will interact with.
* **Lower-level Routines** (`pmrf.math`, `pmrf.rf`). These functions implement simple, standard formulae. They should operate directly on JAX arrays or very simple PyTree structures.
* **RF Models** (`pmrf.models`). These are the main, built-in RF models in the library. They have been separated into an deep organizational structure (e.g. lumped components, surrogates etc.). If a model's forward pass has many different possible implementations (for example, different models for a coaxial line, or different circuit solver algorithms) then separate "solvers" classes should be created alongside the model which implement some abstract interface (referred to as the "strategy" pattern, described below). See `pmrf.models.CoaxialLine` and `pmrf.models.Circuit` for simple and complex examples, respectively.
* **General Algorithms** (`pmrf.optimize` and `pmrf.infer`). These are a combination of functional entry points and wrappers around external libraries. They refer to parametric, model-agnostic algorithms. These modules should be structured in a tiered manner to decouple the Equinox/JAX logic from the RF/model logic. Refer to `pmrf.optimize.base` and `pmrf.optimize.minimize` for a good example.
* **Toolkits** (`pmrf.evaluators`, `pmrf.losses`, `pmrf.likelihoods`, `pmrf.discrepancy_models`, `pmrf.noise_models`, `pmrf.covariance_models`). These are the concrete "glue" components that provide users with a toolkit to mix-and-match together, mostly for power users.
* **Routers** (`pmrf.fitting`) This is an example of a high-level convenience API specifically for fitting models to measured data. Functions here, like `fit`, `fit_minimize` or `fit_sample`, essentially act as "routers", converting different data formats (e.g. `skrf.Network`) and user specs into code that other modules in the library can understand.

### Abstract Classes and the Strategy Pattern

ParamRF makes heavy use of the strategy pattern. Instead of using strings to define the behaviour of functions, classes are used that implement some interface. This allows the library to be extended by end-users, and makes testing and enhancements easier. Equinox makes this pattern very easy. For example, as mentioned above, solvers are *base* classes that contain parameters and implement a relevant interface. Such abstract base classes should have the "Abstract" prefix to make it clear that they define some *interface* as opposed to a concrete implementation.

### Naming Conventions

Alongside those already mentioned, ParamRF follows structured naming conventions:

* Anything that simulates a problem, or maps inputs to outputs, is a "solver", and should be referred to as such. If a specific type of a solver has a well-defined name, e.g. an "optimizer", this name can be used for the class name, but user-facing variables should still be called "solver".
* Parameter names should avoid underscores where possible, grouping similar concepts into single string-names such as "lengthscale" or "epr" (instead of "length_scale" or "ep_r").
* Class names should be as specific as possible, though can be shortened if the concept is ubiquitous within the namespace it lives in. For example, many of the optimizers are just named e.g. NelderMead, whereas simulation solvers which have no ubiqitous domain name or author are defined based on their behaviour e.g. GlobalScatteringCircuitSolver.

## JAX and Equinox "Gotchas"

* **Immutability:** `Model` classes (and dummy structures) inherit from `equinox.Module`. They are therefore immutable and classes are structured around this.
* **Pure Functions:** JAX requires functions to be pure (no side effects) to be compiled with `@jax.jit`. Make sure to use the `jax.lax` module where possible.
* **Vmapping:** The decision was made that user-facing functions should not have to be vmapped (e.g. Model.s), however any internal solvers and algorithms should be writting without vectorization and later "vmapped" using JAX. For example, the circuit simulation algorithms. However, this decision was made later in the library's implementation, and may not yet be fully implemented throughout.
* **Shapes and Types:** The library uses (nfreq, nports, nports) throughout RF functions, but swaps to nfreq on the last axis for statistical functions. This enables easy batching using `jax.vmap`.

## Submitting a Pull Request

1. Create a new branch from `main` (e.g., `git checkout -b feature/new-solver`).
2. Write tests for your new feature or bug fix.
3. Ensure the entire test suite passes (`pytest`).
4. Push your branch and open a Pull Request on GitHub. 

Provide a clear description of what the PR does. If it fixes an open issue, link to it in the description!