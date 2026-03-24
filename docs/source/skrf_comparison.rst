scikit-rf Comparison
=====================
ParamRF is designed to complement scikit-rf and not replace it. A very brief summary of the differences, strengths and weakness of the libraries are presented below.

Core Philosophy
---------------
``scikit-rf`` is fundamentally built around the static ``skrf.Network`` class. While this is excellent for manipulating measurement data (e.g. for calibration) or for generating simple circuits at a given frequency (like capacitors or transmission lines), it handles parameters in a purely functional way. For complex, deeply nested architectures, it becomes tedious to manually propagate variables down through multiple layers of sub-circuit functions in order to arrive at a final network. Because of this functional focus, any analysis or control over the parameters, such as deciding which parameters should be fixed or defining bounds, is left to the user. This is not only tedious, but can add lots of overhead (the author knows!) since the control logic is ultimately in Python.

``ParamRF`` is built around the parametric ``pmrf.Model`` class (:class:`pmrf.Model`). This provides an object-oriented scaffold, where model variables (like R, L, C, physical dimensions etc.) are managed by the framework via `parax.Parameter <https://gvcallen.github.io/parax/api/#parax.Parameter>`_. Parameter names, as well as arrays, bounds, distributions, and other metadata, are automatically available and updated across any complex hierarchy of sub-models, making it trivial to update an entire nested system. This makes fitting, sampling or any type of parametric model analysis very easy!

Concrete Comparison
--------------------
In ``scikit-rf``, as a consequence of the core class being a ``skrf.Network`` (which holds its static S-parameters), scikit-rf uses a helper class ``skrf.Media`` which defines some propagation media. Transmission lines and components are then created **from** this media using e.g. ``res = media.resistor(10)`` or ``line = media.line(10, unit='m')``. This makes components (such as resistors) feel like "second-hand" citizens - the resultant ``res`` and ``line`` objects in these examples are now just ``skrf.Network`` objects, and have lost their original meaning and parameters.

The scikit-rf approach is somewhat in contrast to traditional circuit modelling packages, where the user thinks of the resistor *itself* as the first-class citizen, which is then "embedded" in some media or circuit topology. ``ParamRF`` takes this approach. Components *themselves* are models, such as ``pmrf.models.Resistor`` or ``pmrf.models.PhysicalLine``. The user then instantiates these *directly* using e.g. ``res = Resistor(10)`` or ``line = PhysicalLine(length=10)`` and combines them directly. These not only feels more intuitive, but also allows for more complicated modelling, model simplification (for example, a ``Circuit`` within a ``Circuit`` is just one big circuit and matrix inverse need not occur twice) and model manipulation (varying parameters after the model is constructed etc.).

When to use scikit-rf
---------------------
If your workflow involves standard microwave engineering tasks or static measurements, scikit-rf remains the go-to choice. It excels at:

- **Data I/O & Calibration**: Reading Touchstone files, applying VNA calibrations (TRL, SOLT), de-embedding network data.
- **Static Network Manipulation**: Cascading measurements, port impedance transformations, and standard plotting (e.g., Smith charts).
- **Static Network Analaysis**: Analyze existing networks, such as using time-domain reflectometry or checking networks for passivity/causality.
- **Flat Circuit Generation**: Quickly generating analytical components or simple, flat circuit models where continuous, high-dimensional parameter optimization or analysis is not the primary goal.

When to use ParamRF
-------------------
If your workflow requires the use of complex or high-dimensional RF models, ParamRF provides the architecture required to do this both conveniently and efficienctly. Further, some of the built-in backends might make it a preference even for simple circuit modeling tasks. It excels at:

- **Deeply Nested Modeling**: Building complex, hierarchical models without the boilerplate of manually passing variables through sub-circuit functions.
- **Analytical and Numerical Modeling**: The ``Model`` class is easy to extend to implement custom equation-based models. Further, since ParamRF integrates with the machine-learning focused library ``Equinox``, building numerical and surrogate RF models is native.
- **High-Speed Fitting**: Leveraging JAX Just-In-Time (JIT) compilation to fit models to target data is much faster than standard Python loops.
- **Sampling and Active Learning**: Randomly and adaptively sampling an arbitrary, expensive RF model such as an external EM simulation. 
- **Statistical Analysis**: Running automated Monte Carlo or Latin Hypercube sampling across your model's entire parameter space.

How They Work Together
----------------------
In a practical fitting workflow, both ``scikit-rf`` and ``ParamRF`` are used:

- Use ``scikit-rf`` to load your raw VNA measurements and perform TRL de-embedding.
- Define your theoretical, deeply nested circuit topology using :class:`pmrf.Model` and the models library in :mod:`pmrf.models`.
- Use a solver such as :class:`pmrf.optimize.ScipyMinimizer` to optimize your model's parameters to the measured scikit-rf ``Network``.

Further, ``pmrf.Model`` can easily be converted to ``skrf.Network`` by simply passing a frequency using ``pmrf.Model.to_skrf(freq)``, so you can easily define your model in ``ParamRF`` and then do any further analysis in ``scikit-rf``!