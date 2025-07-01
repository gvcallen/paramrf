# ParamRF: Parametric Microwave Circuit Modelling, Fitting and Sampling

## Overview

ParamRF, or `pmrf`, is an easy-to-use framework built on top of `jax`, used for describing general microwave circuit models in an efficient, parametric, object-orientated manner.

The framework integrates concepts from several pacakages into one, such as `scikit-rf` for general syntax and RF concepts, `equinox` for model building, and `jax` for high-speed, hardware-accelerated calculations and automatic differentiation. The long-term goal is to allow the building of full, complex circuit topologies, however currently only simplified models are available.

`pmrf` provides a declarative interface that easily compiles models into an efficient linear algebra graph using `jax`, and also provides the ability to optimize these models using commonly available or custom-defined fitting algorithms. An introduction into the core concepts, as well as the showcase of some simple examples, is provided below.

## Core Concepts

The library revolves around a few key building blocks:

* **`pmrf.Model`**: This represents the base class for any computable RF component, such as foundational models (resistors, transmission lines etc.) or complex circuit models. Compared to `skrf.Network`, models are *functional* in nature, meaning that they only store their representation (parameters and computation) as opposed to their *data*. Therefore, all model properties such as `s`, `a` etc. accept frequency as an input. Models are composable, meaning you can connect them together using simple operators (`**` for cascade/series, for example) to create more complex models.
* **`pmrf.Parameter`**: A `Parameter` is `pmrf`'s representation of a circuit component parameter with additional metadata. It allows for bounds, can be marked as `fixed`, and can have a statistical prior associated with it for Bayesian fitting. Any attribute of a `Model` can be a `Parameter`, alongside other sub-models.
* **`pmrf.Frequency`**: This represents a JAX-compatible object that defines the frequency axis over which your models are evaluated. As mentioned, this is one of the main differences compared to `scikit-rf` i.e. the frequency object is an *input* to the model as opposed to an attribute of the model itself. This allows for the decoupling of model evaluations and parameterization, and also provides the ability to automatically differentiate with respect to frequency.
* **`pmrf.fitting`**: This is the fitting module, which contain various "Fitter" classes that each take a model and `skrf.Network` measurement data as input, as well as some fitting hyperparameters (such as the features and a definition for the cost function, if desired), and fits the model parameters.

## Model Definition and Composition

Models are defined declaratively by inheriting from `pmrf.Model` and specifying parameters and sub-models as class attributes. The framework's compositional nature allows for complex systems to be constructed from simpler, validated components.

The following example demonstrates the definition of a non-ideal resistor model, composed of an ideal resistor and a parasitic Pi-network.

```python
import pmrf as prf
from pmrf.models import Resistor, PiCLC

# Define a new model class inheriting from prf.Model.
# Its attributes are sub-models and parameters that constitute its structure.
class MyNonIdealResistor(prf.Model):
    # Define sub-components. Default values can be provided.
    # `res` represents the ideal resistive element.
    res: Resistor = Resistor(R=100.0)
    
    # `parasitics` represents the unwanted capacitive and inductive effects.
    parasitics: PiCLC = PiCLC(C1=0.05e-12, L=0.1e-9, C2=0.05e-12)

    # The model's behavior is defined by implementing a primary network matrix 
    # function, such as `a` for the ABCD-matrix.
    def a(self, freq: prf.Frequency):
        # The `__pow__` (**) operator is overloaded to represent a `Cascade` of models.
        # `combined_model` below will be of type `pmrf.models.containers.Cascade`.
        combined_model = self.parasitics ** self.res
        return combined_model.a(freq)

# --- Using the model ---

# Instantiate the composite model.
my_resistor = MyNonIdealResistor()

# The `with_params` method returns a new, updated model without mutating the original.
# Parameters can be addressed by their hierarchical names.
my_resistor = my_resistor.with_params(parasitics_C1=0.08e-12)

# Define a frequency axis for evaluation.
freq = prf.Frequency(start=1, stop=18, npoints=201, unit='ghz')

# Evaluate the model's S-parameters.
s_params = my_resistor.s(freq)

print(f"Evaluated S-parameter matrix shape: {s_params.shape}")

# For analysis and plotting, convert the evaluated model to a scikit-rf Network.
ntwk = my_resistor.to_skrf(freq)
# ntwk.plot_s_db(...)
```


## Model Fitting

A primary application of `pmrf` is the optimization of model parameters to align with measured data. The fitting module provides a unified interface to perform this task using various numerical methods.

The general workflow consists of defining a parametric model, loading empirical data, configuring a fitter, and executing the optimization routine.

#### Available Fitters:

* **`FrequentistFitter` / `ScipyFitter`**: Provides access to gradient-based and gradient-free optimization algorithms from the `scipy.optimize` library. These are used to find a single point estimate of the parameter values that best minimizes a given cost function.
* **`BayesianFitter` / `PolyChordFitter`**: Enables Bayesian inference through dynamic nested sampling. This approach yields not only optimized parameter values but also their full posterior probability distributions and the Bayesian evidence, which is crucial for model comparison and uncertainty quantification.

#### Fitting Example

The following provides a toy template for a typical fitting process.
See the "examples" folder in this reposistory for a more realistic script.

```python
import skrf as rf
import pmrf as prf
from pmrf.models import CLCResistor, Resistor, PiCLC

# 1. Load empirical measurement data into a scikit-rf Network object.
try:
    measured_ntwk = rf.Network('my_device_measurement.s2p')
except FileNotFoundError:
    print("Skipping fitting example: data file not found.")
    # In a real scenario, you would handle this error.
    measured_ntwk = None

if measured_ntwk:
    # 2. Instantiate the parametric model to be fitted.
    # Initial values serve as the starting point for the optimization.
    model_to_fit = CLCResistor(
        res=Resistor(R=50.0),
        clc=PiCLC(C1=0.1e-12, L=0.2e-9, C2=0.1e-12)
    )

    # Parameters can be excluded from optimization by marking them as `fixed`.
    # For this example, we will fit all default (non-fixed) parameters.
    # See the documentation for more details.

    # 3. Configure the fitter, which encapsulates the model, data, and cost function.
    fitter = prf.fitting.ScipyFitter(
        model=model_to_fit,
        measured=measured_ntwk
        # A custom cost function and features can optional be provided
        # (e.g., 's21_db', 'a11_deg' etc.) to use for fitting.
        # The default is 's11' with an L2 norm in dB
    )

    # 4. Execute the fitting routine, passing parameters along to scipy.
    fit_result = fitter.run(method='SLSQP')

    # The result object contains the optimized model and detailed fit metrics.
    print("Optimization Complete.")
    print("Optimized Parameters:")
    for name, param in fit_result.model.params.items():
        print(f"  {name}: {param.value:.3e}")
```

## Key Features

* **JAX Backend**: Leverages `JAX` for Just-In-Time (JIT) compilation of models to high-performance hardware (CPU, GPU, TPU). This removes python overhead from context switch, enables better vectorization an parallelization, and provides automatic differentiation through the entire model structure, enabling more effective gradient-based optimization.
* **Parametric & Composable Design**: Models are defined declaratively as `Equinox` modules, allowing for the natural composition of complex systems from simpler sub-models. Any model attribute can be either a `Parameter` or another `Model`, allowing flexible, hierarchial model building.
* **`scikit-rf` Integration**: Designed for seamless interoperability with `scikit-rf`. `pmrf` models can be evaluated and converted to `skrf.Network` objects, providing access to `scikit-rf`'s library of analysis and plotting tools.
* **Advanced Fitting Engine**: Offers a unified interface for both frequentist optimization and advanced Bayesian inference.
* **Extensibility**: Designed to be extendable, such that additional models, fitting algorithms, cost functions, sampling routines etc. can easily be implemented.
