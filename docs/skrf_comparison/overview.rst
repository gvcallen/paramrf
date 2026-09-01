Overview
=====================
**ParamRF** is designed to complement **scikit-rf** and not replace it, offering a fundamentally different approach to RF modeling. A very brief summary of the differences, strengths and weakness of the libraries is presented below.

Core Philosophy
---------------
scikit-rf is built around the static ``skrf.Network`` class. While this is excellent for manipulating measurement data (e.g. for calibration) or for generating simple circuits at a given frequency, its structure is such that parameters are simply inputs to functions. For complex, deeply nested architectures, it becomes tedious to manually propagate variables down multiple layers of sub-circuit functions in order to arrive at a final network.

Because of this approach, any analysis or control over parameters, such as deciding which should be fixed, adding constraints for optimization etc., is left to the user. This is not only tedious, but can add lots of overhead (the author knows!) since the control logic is ultimately in Python. Such overhead can reduce performance both of once-off circuit simulations and during optimization.

ParamRF is built around the parametric :class:`pmrf.Model` class. This provides an object-oriented scaffold, where model parameters are stored internally, as opposed to their S-parameter matrix and frequency. Parameter bounds, constraints etc. are then stored alongside these parameters and propagated across any complex hierarchy of models, making it trivial to work with deeply nested models. This makes modular fitting, optimization, and sampling much more convenient.

Concrete Comparison
--------------------
In scikit-rf, as a consequence of the core class being ``skrf.Network`` (which holds its static S-parameters), scikit-rf uses a helper class ``skrf.Media`` which defines some propagation media. Transmission lines and components are then created *from* this media using e.g. ``res = media.resistor(10)`` or ``line = media.line(10, unit='m')``. This makes components (such as resistors) feel like "second-class" citizens - the resultant ``res`` and ``line`` objects in these examples are now just ``skrf.Network`` objects storing their S-matrix, and have no access to their original parameters.

The scikit-rf approach is somewhat in contrast to traditional circuit modeling packages, where the user thinks of the resistor *itself* as the first-class citizen. In ParamRF, components *themselves* are therefore the core classes (such as :class:`~pmrf.models.components.lumped.Resistor` or :class:`~pmrf.models.components.lines.PhysicalLine`) and their S-parameters are simply a result of evaluating the model. This is similar to the approach taken in machine learning. Specifically, the user instantiates model instances *directly* (using e.g. ``res = Resistor(10)`` or ``line = PhysicalLine(length=10)``) and combines them to create *new* models. The resultant model outputs (e.g. S-parameters) are then *lazily* evaluated when needed.

This approach not only feels more intuitive, but also allows for more complex modeling, computational simplifications, and model manipulation (i.e. modifying parameters after a model is constructed). Further, since ParamRF builds on top of the ``jax`` library (instead of ``numpy``), models can be just-in-time compiled, as well as differentiated with respect to either parameters or frequency using *auto-differentiation*.

When to use scikit-rf
---------------------
If your workflow involves standard microwave engineering tasks or static measurements, scikit-rf remains the go-to choice. It excels at:

- **Data I/O & Calibration**: Reading Touchstone files, applying VNA calibrations (TRL, SOLT), de-embedding network data.
- **Static Network Manipulation**: Cascading measurements, port impedance transformations, and standard plotting (e.g., Smith charts).
- **Static Network Analysis**: Analyzing existing networks, such as using time-domain reflectometry or checking networks for passivity/causality.
- **Flat Circuit Generation**: Quickly generating analytical components or simple, flat circuit models where continuous, high-dimensional parameter optimization or analysis is not the primary goal.

When to use ParamRF
-------------------
If your workflow requires the use of complex or high-dimensional RF models or state-of-the-art optimization techniques, ParamRF provides the architecture to do this both conveniently and efficiently. It excels at:

- **Deeply Nested Modeling**: Building complex, hierarchical models without the boilerplate of manually passing variables through sub-circuit functions.
- **Differentiability**: Using JAX, models can be differentiated with respect to both parameter and frequency.
- **Analytical and Numerical Modeling**: The :class:`pmrf.Model` class is easy to extend to implement custom equation-based models. Further, since ParamRF integrates with the machine-learning focused library ``Equinox``, numerical/surrogate/ML models can be built natively.
- **High-Speed Fitting**: Leveraging JAX Just-In-Time (JIT) compilation to fit models to target data is much faster than standard Python loops.
- **Statistical Analysis**: Running automated Monte Carlo or Latin Hypercube sampling across your model's entire parameter space.

Also note that :class:`pmrf.Model` instances can easily be converted to ``skrf.Network`` objects by simply passing a frequency to :meth:`pmrf.Model.to_skrf`, allowing, for example, models to be defined and optimized in ParamRF and then further analyzed in scikit-rf.
Physics validation against scikit-rf
------------------------------------
The two libraries implement the same published transmission-line formulations independently, which makes scikit-rf a useful external reference for ParamRF's physics. ParamRF's coaxial and microstrip lines are checked against ``skrf.media.Coaxial`` and ``skrf.media.MLine`` over a curated wideband matrix, from kilohertz frequencies through tens of gigahertz, covering per-unit-length R, L, G and C, characteristic impedance, propagation constant and S-parameters. The matrix lives in ``tests/test_models/test_lines_skrf_matrix.py``, where each case records its purpose and its own tolerances.

scikit-rf is treated as guidance, not as ground truth. Where the two disagree, the cause is traced to a modelling choice on one side or the other and recorded in the test, rather than absorbed into a wider tolerance.

**The complex-permittivity convention only.** ParamRF carries permittivity complex throughout, matching the convention used by ADS and AWR. scikit-rf's ``MLine`` additionally offers a QUCS compatibility mode, which uses a real permittivity for the quasi-static and dispersion stages. ParamRF does not implement that path, so QUCS compatibility is out of scope: the comparisons use ``compatibility_mode=None`` exclusively.

Stripline has no scikit-rf counterpart, and is instead validated against Cohn's published formulation and against analytic identities.
