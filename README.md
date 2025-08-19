# ParamRF: Parametric Microwave Circuit Modelling, Fitting and Sampling

**ParamRF**, or ``pmrf``, is a declarative circuit modelling framework catering for the frequency-domain fitting and simulation of (microwave) circuit models in an efficient, parametric, object-orientated manner. The framework integrates concepts from several packages into one, such as `scikit-rf` for general syntax and RF concepts, `equinox` for model building, and `jax` for high-speed, hardware-accelerated calculations and automatic differentiation.

| **ParamRF** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/paramrf/paramrf](https://github.com/paramrf/paramrf) |
| **Docs** | [paramrf.github.io/paramrf](https://paramrf.github.io/paramrf) |

## Key Features
* **Declarative and Composable Modelling**: Allows for the definition of models using either a self-documenting, declarative syntax or via compositional techniques such as cascading. Since models can consist of a mix of ``Parameter`` objects as well as other ``Model``'s, this allows for a natural means of building complex, hierarchial models from both equations and other sub-models.
* **Unified Fitting Engine**: Provides a number of commonly available fitting algorithms with a unified interface, catering for both classical frequentist optimization and statistical Bayesian inference.
* **JAX Backend**: Leverages `JAX` for Just-In-Time (JIT) compilation of models to high-performance hardware (CPU, GPU, TPU). This removes python overhead due to interpreter context switching; enables better vectorization and parallelization; and provides automatic differentiation through the entire model structure, enabling new analysis and more efficient gradient-based optimization.
* **Extensibility**: Designed to be extendable, such that additional models, fitting algorithms, cost functions, sampling routines etc. can easily be implemented.
* **scikit-rf Integration**: Designed for seamless interoperability with *scikit-rf*, ``pmrf`` models can be evaluated and converted to ``skrf.Network`` objects, providing access to *scikit-rf*'s library of analysis and plotting tools.

## Installation
ParamRF can be installed using pip directly from the github page:

``
pip install git+https://github.com/paramrf/paramrf@main
``

Several additional (optional) dependency packs can also be installed instead of manually installing the packages.

For Polychord fitting:
``
pip install 'paramrf[polychord] @ git+https://github.com/paramrf/paramrf@main'
``

For blackjax fitting:
``
pip install 'paramrf[blackjax] @ git+https://github.com/paramrf/paramrf@main'
``