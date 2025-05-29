# ParamRF: Parametric Microwave Circuit Modelling, Fitting and Sampling

## Overview

This package provides the ability to describe RF circuit models in an object-orientated, parametric manner, acting as an extension of the scikit-rf library. It allows for circuit modelling, fitting and sampling in a convenient manner. The library mainly builds on top of packages scikit-rf, scipy, polychord, and, of course, numpy. It is, however, very much still in beta

## Examples

Currently there is only one example, demonstrated in the script in `examples/fit_cable.py`, which shows how top fit the 10 meter cable's S-parameters using the built-in PhysicalCoax model and a frequentist SLSQP solver.


## Documentation
The initialize functions for many classes have some introductory docs. This section simply explains the organization of the package and its sub-modules, to provide a higher-level overview.

### _core_ module
This module contains the core object-orientated classes. These are modifications of the scikit-rf `Network` class, with the ability for networks to be dependent via the `ObservableNetwork` class; to update based on a set of parameters via the `ParametricNetwork` class; or to be composed by another list of networks via the `CompositeNetwork` class.

### _modelling_ module
This module contains various common circuit models and elements, as well as the main `NetworkSystem` class, which represents a collection of these models. 
In the _elements_ sub-module:
- _Lumped_ elements that are not already present in scikit-rf itself are in `elements/lumped.py` (e.g. an ideal tranformer).
- So-called _topological_ elements (e.g. pi-junctions, T-networks etc.) are present in `elements/topological.py`.
- _Distributed_ elements, currently only the foundational RLGC transmission line, are found in `elements/distributed.py`.

In the _models_ sub-module:
- General transmission line models (e.g. for coaxial and microstrip lines) are found in `models/lines.py`.
- Physical/non-ideal element models (e.g. for a physical resistor) are in `models/physical.py`
- Models for connectors are in `models/connectors.py`.

### _statistics_ module
This module contains everything to do with data and statistical model evaluation, such as generating a feature matrix from Networks, evaluating priors and likelihoods, and storing a list of parameters.
- The `Feature` class in the _evaluation_ sub-module is designed to extract features from a scikit-rf `Network` class, such as S11 magnitude, S21 real, etc. Of importance is the `extract_features` function, which extracts features from multiple networks and creates a "feature matrix" containing all feature values across frequency.
- The `Prior` and `Likelihood` classes encapsulate statistical prior and likelihood functions. They are used to allow the `NetworkFitter` class to easily sample from the parameter space and compute log-likelihood values (for bayesian solvers).
- The `Modifier` and `ModifierChain` classes encapsulate the idea of performing a series of modifications on a numpy array. In practice, this is used on the feature matrix to easily define different types of cost functions (for frequentist solvers) in a modular way.
- The `ParameterSet` class derives from a pandas `DataFrame` and contains all the parameter read/write code. It also allows for _derived_ parameters, where one parameter can be set equal to another.

### _fitting_ module
This module contains the main _NetworkFitter_ class, as well as various other helper classes:
- The `NetworkFitter` class is the main coordinator. It contains references to the fitting parameters, the models in the fit (itself represented in the generic `NetworkSystem` class), and various settings. It is the class that ultimately runs the optimization/fit.
- The `Target` class is the grouping of a circuit model and its measured data. It also allows for computing the target's cost and likelihood function.

### _plotting_ module
This module contains functionality to plot common plots, such as S11 parameters, gains etc. It is encapsulated in a `Plotter` class to de-couple it from the fitting module.

### _rf_ module
This module contains RF computations ,such as the actual calculation of available gains, and the concatenation of two-port networks.

### _misc_ module
This module contains any misc functionality, such as mathematical functions etc.
