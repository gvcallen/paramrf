---
title: 'ParamRF: A Modern Framework for Parametric Radio Frequency Modeling'
tags:
  - Python
  - JAX
  - radio frequency
  - circuit modeling
  - optimization
  - Bayesian inference
authors:
  - name: Gary V. C. Allen
    orcid: 0009-0005-6572-598X
    affiliation: 1
  - name: Dirk I. L. de Villiers
    affiliation: 1
    orcid: 0000-0003-1273-5365
affiliations:
 - name: Stellenbosch University, South Africa
   index: 1
date: 17 May 2026
bibliography: paper.bib
---

# Summary

ParamRF is an open-source Python framework for the efficient, parametric modeling of radio frequency (RF) circuits and surrogates. The library provides a declarative, object-oriented API, allowing for the composition of deep, nested circuits and the definition of custom models via inheritance. Building on top of the JAX computational ecosystem [@jax2018], models are represented as pure functions with immutable data alongside. This architecture allows the library to make use of JAX's just-in-time (JIT) compilation to modern hardware (CPU, GPU, TPU), as well as its vectorization and automatic differentiation transformations. Using ParamRF's built-in minimization and Bayesian sampling modules, this enables high-performance optimization, advanced parameter extraction, and new design opportunities.

# Statement of Need

The use of radio frequency circuit models is central to fields such as high-frequency electronics, power systems, and semiconductor design. A core engineering requirement in these domains is the creation of parametric circuit models that must be optimized to satisfy specific design goals or fit to measured data for calibration purposes. ParamRF addresses this need by providing researchers and RF engineers with a flexible, programmatic Python framework to easily compose, differentiate, and optimize these circuits.

# State of the Field

Currently, researchers and RF engineers often rely on commercial, GUI-driven tools. While powerful, these tools typically lack a flexible programmatic interface, making it difficult to define custom error functions, automate routines, or integrate with modern statistical solvers. Conversely, standard script-based approaches (often built on libraries like NumPy) can introduce significant overhead due to interpreter context switching and dynamic memory allocation. Further, they lack the ability to compute exact analytical gradients, forcing optimizers to rely on slower and approximate numerical differentiation, and also cannot natively encapsulate circuit model uncertainty using Bayesian methods.

Within the open-source community, `scikit-rf` [@scikit-rf] serves as an industry standard for microwave and RF network creation and analysis. However, its primary design is data-driven, storing S-parameter matrices and frequency arrays directly. ParamRF is designed to complement and not replace `scikit-rf` by filling the gap for parameter-focused optimization. It provides an architecture where the core primitives are parametric models as opposed to matrix data containers, allowing the easy creation of complex hierarchical circuits which can be compiled, differentiated and optimized. This approach is not only more intuitive and efficient for optimization purposes, but also provides a foundation for new circuit design opportunities. Models can also optionally be converted to static `scikit-rf` networks after simulation in ParamRF, providing integration with the rest of the Python RF ecosystem.

# Software Design

ParamRF uses a unique approach to circuit modeling via a functional programming paradigm within an object-oriented syntax:

- **JAX-Native:** The core primitives, `pmrf.Model` and `pmrf.Frequency`, are both JAX PyTrees and Python dataclasses. This means that models are strictly immutable as well as "lazy" i.e., they do not store evaluated matrices, but rather define the pure functions and parameters used to compute their network responses. This provides full compatibility with the JAX ecosystem and therefore the XLA (Accelerated Linear Algebra) compiler.
- **Declarative Syntax:** Users can define deep, hierarchical structures through a self-documenting syntax. Models can be easily nested and combined using either operator overloading (e.g., using the `**` operator for two-port cascading) or the `pmrf.models.Circuit` class for general circuit topologies. For custom models, `pmrf.Model` can also be overriden where parameters can be given default constraints, catering for more advanced use-cases.
- **Autodifferentiation:** Because the framework is built on JAX, exact mathematical derivatives of circuit responses (such as S-parameters) can be calculated with respect to both frequency and component parameters. This provides gradient-based solvers with exact Jacobians, greatly improving performance and stability for high-dimensional models. Gradients can also be employed for advanced circuit analysis and design purposes, which is an active area of research.
- **Parameter Constraints and Manipulation:** Parameters can be scaled, constrained, tied together, or assigned prior probability distributions. This relies on a concept known as "unwrapping". For example, the built-in factory functions in `pmrf.parameters` return nested wrappers that are unwrapped automatically at evaluation time. This approach provides a bridge between JAX's pure functional style, and the declarative, object-oriented syntax desired by RF engineers.
- **Built-in Solvers:** The framework provides high-level sub-modules (`pmrf.optimize`, `pmrf.infer`, `pmrf.fitting`) that abstract different solver backends into a unified interface. Users can easily swap between classical optimization (using SciPy or Optimistix) and Bayesian inference (using BlackJAX) to perform parameter estimation and uncertainty quantification.

# Research Impact

ParamRF was initially developed for the REACH (Radio Experiment for the Analysis of Cosmic Hydrogen, [@de2022reach]) collaboration. REACH is one of several forefront experiments attempting to detect the cosmological global 21 cm signal (notable others are EDGES [@bowman2018absorption] and SARAS [@patra2013saras]). The project employs physically motivated modeling techniques to perform high-precision calibration of their radiometric instrument, and utilizes ParamRF for the circuit modeling backend in their pipelines. This novel circuit-based calibration approach has been published in the Journal of Astronomical Instrumentation [@allen2026circuit] and will be presented at the Global 21 cm Workshop 2026 [@globalworkshop2026]. Implementing these methods using existing solutions proved inefficient and lacked the capacity for Bayesian uncertainty quantification. By leveraging the automatic differentiation and just-in-time compilation provided by ParamRF, the project achieved between 18x and 340x faster optimization of their balun circuit [@allen2026efficient]. Further, the pipelines rely on the Bayesian solvers natively available in ParamRF, providing a foundation for ongoing statistical research.

Although ParamRF originated in radio astronomy, the demand for programmatic and differentiable RF modeling is broad, and these optimization speedups are widely applicable across high-frequency engineering. For instance, recent work on non-uniform amplifier matching networks [@shila2025computationally] required implementing design equations directly in Julia to utilize automatic differentiation. ParamRF eliminates the need for such single-use implementations by generalizing this capability to arbitrary circuit topologies. This allows for more efficient optimization, broader design space exploration, and faster calibration workflows directly within Python.

Finally, ParamRF significantly lowers the barrier to entry for sensitivity-driven design and microwave circuit uncertainty quantification. While traditional optimizers provide single-point estimates, ParamRF’s native integration with Bayesian sampling allows for full design space encapsulation. The ability to compute exact mathematical derivatives, over and above providing exact gradients to solvers, also unlocks exciting new opportunities for advanced circuit analysis and research.

# Example Usage

The following snippet demonstrates ParamRF's syntax and optimization API. A standard RLC (resistor-capacitor-inductor) circuit is first composed using operator overloading. Then, an $S_{11}$ reflection design goal is defined, as well as the target frequency passband over which it should be met. Finally, the goal is minimized, and the initial and optimized models are plotted against each other over a wider frequency range.

```python
import pmrf as prf
from pmrf.models import Resistor, Inductor, Capacitor

R = prf.Unconstrained(50.0)
L = prf.Bounded(0.0, 100.0, scale=1e-9)
C = prf.Bounded(0.0, 100.0, scale=1e-12)

model = Resistor(R) ** Inductor(L) ** Capacitor(C)
goal = prf.evaluators.Goal('s11_db', '<', -20)
passband = prf.Frequency(2, 5, 101, 'GHz')

result = prf.optimize.minimize(
  objective=goal,
  model=model,
  frequency=passband,
  solver=prf.optimize.NelderMead()
)

plot_freq = prf.Frequency(1, 6, 101, 'GHz')
model.plot_s_db(plot_freq, m=0, n=0, label='initial')
result.model.plot_s_db(plot_freq, m=0, n=0, label='optimized')
```

![Optimized vs Initial RLC.](rlc.png){width=65%}

# AI Usage Disclosure

We disclose the use of generative AI during the development of this project:

* **Tool Identification:** Google Gemini 2.1 and 3.1 Pro was the primary tool used. [@gemini2023]
* **Scope of Assistance:** The tool was used to automate repetitive tasks, locate and fix bugs, scaffold documentation and tests, and generate the initial implementation for some lower-level models and solver wrappers.
* **Human Review:** All AI-generated content was thoroughly reviewed and validated by the authors. All primary architectural and design decisions were made exclusively by the authors, who bear responsibility for the accuracy and originality of the submitted materials.

# Acknowledgements

The research was supported by the South African Radio Astronomy Observatory, which is a facility of the National Research Foundation, an agency of the Department of Science and Technology (Grant Number 75322). The authors would also like to thank the Kavli Foundation, and the Science and Technology Facilities Council, grant number EP/Y02916X1/1, for supporting the REACH project, as well as the developers of JAX, Equinox [@kidger2021equinox], scikit-rf, and the broader open-source scientific Python community that made this framework possible.

# References