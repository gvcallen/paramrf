Getting Started
=====================

**ParamRF** provides a declarative modelling interface that compiles RF models, such as circuit models, using *JAX*. This page provides in introduction into how to define models, as well as fit them.

Core Concepts
~~~~~~~~~~~~~~~~~~~~

The library revolves around a few key building blocks:

* :class:`pmrf.Model`: The base class for any RF model. When inherited from, methods such as *s*, *a*, *z* and *y* can be overriden to define model S-parameters, ABCD-parameters etc. These methods all accept frequency as input. On the other hand, *__call__* can be overridden to return a model instance itself, for more complex compositional model building.
* :class:`pmrf.Frequency`: A wrapper around a JAX array that defines the frequency axis over which models are evaluated.
* :class:`parax.Parameter`: A parameter in a model (from `parax <https://github.com/gvcallen/parax>`_), storing its value and metadata. This allows for parameter bounds and scaling, marking parameters as *fixed*, and associating a *distribution* with the parameter for Bayesian fitting.
* :class:`pmrf.Evaluator`: A lower level object that is used to extract features from a model in a composable manner. These are created under-the-hood e.g. when specifying custom features for fitting, but can also be specified directly for advanced cost and likelihood functions.

Model Composition
~~~~~~~~~~~~~~~~~~~~
**ParamRF** provides a component library with commonly-used models such as lumped and distributed elements. Models can be built directly using these in a compositional approach.

Cascaded Models
^^^^^^^^^^^^^^^^^^^
For simple circuit element chains, the ** operator can be used to cascade several models together.

The example below creates an RLC filter and terminates it in an open circuit. The resultant ``rlc`` is a first-class :class:`pmrf.Model` of type :class:`pmrf.models.Cascade`, consisting of parameters representing the respective *R*, *L* and *C* parameters. The S11 is then plotted using ``scikit-rf`` under the hood.

.. literalinclude:: ../../../examples/1_rlc_cascade.py
   :language: python

Circuit Models
^^^^^^^^^^^^^^^^^^^
For complex circuits, ParamRF offers the ability to combine models in any desired configuration using the :class:`pmrf.models.Circuit` class. This class accepts a list of "connections". Each entry in this list is a node in the circuit. Each node is another list, with each element being a tuple for each connected circuit element or sub-model. Each tuple then contains the model object, as well as the index of the port for that model that is connected in that node.

The following example uses this method to define a two-port PI-CLC network. "External" nodes (each entry in the outer list) are numbered as E0, E1 etc. whereas "internal" port indices (ports for each model in the circuit) are numbered per element as I0, I1 etc. The model is then converted to a scikit-rf network and plotted.

.. image:: circuit_clc.png
   :alt: pi-CLC circuit diagram
   :width: 600px
   :align: center

.. literalinclude:: ../../../examples/2_clc_circuit.py
   :language: python

Model Inheritance
~~~~~~~~~~~~~~~~~~~~
For more complex models (such as equation-based ones), users can inherit directly from the :class:`pmrf.Model` class and override one of the network properties (such as ``s``, ``a``, or ``y``) or the ``__call__`` method.

Any attributes of a model are classified as either *static* or *dynamic*. By default, fields of built-in types such as ``str``, ``int``, ``list`` etc. are seen as static in the model hierarchy, whereas those annotated as a :class:`parax.Parameter` or :class:`pmrf.Model` are dynamic and can be adjusted (for example, by fitting routines).

Note that parameter initialization is flexible: parameters may be populated with a simple float value; using factory methods such as :class:`parax.Uniform`, :class:`parax..Normal` or :class:`parax..Fixed`; or directly using the :class:`parax.Parameter` class constructor.

Equation-based Models
^^^^^^^^^^^^^^^^^^^^^
The following example demonstrates custom model definition by defining a capacitor from first principles. This could be used, for example, to implement more complex analytic or surrogate models. Here, one of the typical network properties, such as :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, or :meth:`pmrf.Model.z`, must be overriden, returning the resultant matrix directly.

.. literalinclude:: ../../../examples/3_equation_capacitor.py
   :language: python

Circuit Models
^^^^^^^^^^^^^^^^^^^
Sometimes it is still convenient to inherit from :class:`pmrf.Model` while still building the model using cascading or :class:`pmrf.models.Circuit`. In this case, the model can be built from sub-models fields/attributes, and returned by overriding the :meth:`pmrf.Model.__call__` method.

The following example creates a PI-CLC model once again, but using the above method. Note how certain parameters can be given initial parameters, bounds or fixed to a constant (useful for fitting).

.. literalinclude:: ../../../examples/4_clc_inheritance.py
   :language: python

Fitting
~~~~~~~~~~~~~~~~~~~~

Models can easily be fit using to measured data using the :mod:`pmrf.fit` module. The general workflow consists of defining a model, loading data via *scikit-rf*, initializing the solver (optimizer/inferer), defining the features to optimize (e.g. S11), and running the fit.

Solvers
^^^^^^^^^^^^^^^^^^^^

ParamRF allows for optimization using either ``scipy.optimize.minimize`` or ``optimistix.minimise``, and Bayesian inference using ``inferix``, which provides wrappers for ``Polychord`` and ``BlackJAX``.

* *Scipy*: Provides a wrapper around gradient-based and gradient-free optimization algorithms from ``scipy.optimize`` in :class:`pmrf.optimize.ScipyMinimizer`. This includes algorithms such as *SLSQP*, *Nelder-Mead* and *L-BFGS*. These algorithms are host-native and cannot run on the GPU.
* *Optimistix*: Provides JAX-native optimization algorithms, such as ``optimistix.BFGS``. These algorithms run their loop directly in JAX, and therefore can be compiled to any architecture (CPU, GPU, TPU).
* *Inferix*: Enables Bayesian inference through nested sampling and MCMC sampling using e.g. ``inferix.Polychord`` and ``inferix.NUTS``. This approach provides maximum likelihood parameters, as well as full posterior probability distributions and Bayesian evidence for model comparison.

Example
^^^^^^^^^^^^^^^^^^^^

The following provides an example of fitting the built in :mod:`pmrf.models.CoaxialLine` model to the measurement of 10m coaxial cable (provided as an example in the `GitHub <https://github.com/gvcallen/paramrf/tree/main/examples>`_). Data is loaded using ``scikit-rf``, the model is instantiated with appropriate initial parameters, the fit is run, and results are plotted.

.. literalinclude:: ../../../examples/5_fit_cable_scipy.py
   :language: python

.. Sampling
.. ~~~~~~~~~~~~~~~~~~~~

.. ParamRF also provides the ability to randomly or adaptively sample models. The :mod:`pmrf.sample` module provides an interface for this, with simple one-shot sampling algorithms such as *uniform* or *Latin Hypercube*, as well as more advanced adaptive sampling algorithms (such as *uncertainty* sampling) for expensive EM simulations.

.. Main Samplers
.. ^^^^^^^^^^^^^^^^^^^^

.. * :class:`pmrf.sample.UniformSampler`: Uniform sampling.
.. * :class:`pmrf.sample.LatinHypercubeSampler`: Latin hypercube sampling.
.. * :class:`pmrf.sample.EqxLearnUncertaintySampler`: Enables surrogate model uncertainty sampling from ``eqx-learn``. This provides the ability to uncertainty sample using classical machine learning surrogate models, such as Gaussian Processes.

.. Example
.. ^^^^^^^^^^^^^^^^^^^^

.. The below example demonstrates the sampling of 10 different resistor networks with uniform resistance between 9 and 11 ohms.

.. .. literalinclude:: ../../../examples/3_simulate_resistor.py
..    :language: python