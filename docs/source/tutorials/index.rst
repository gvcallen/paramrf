Tutorials
=========

The following tutoral demonstrates describes the optimization and inference interface, and provides a brief example of fitting a model to data. More tutorials will be added soon.


Fitting Example
^^^^^^^^^^^^^^^

The following provides an example of fitting the built in :mod:`~pmrf.models.components.lines.uniform.CoaxialLine` model to the measurement of 10m coaxial cable (provided as an example in the `GitHub <https://github.com/gvcallen/paramrf/tree/main/examples>`_). Data is loaded using ``scikit-rf``, the model is instantiated with appropriate initial parameters, the fit is run, and results are plotted.

.. literalinclude:: ../../../examples/1_fit_cable_scipy.py
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