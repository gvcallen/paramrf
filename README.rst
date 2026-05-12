.. title:: Home

|tests_badge| |docs_badge|

.. image:: https://raw.githubusercontent.com/gvcallen/paramrf/main/assets/logo.png
   :align: center
   :alt: ParamRF Logo


**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modeling framework. It provides a declarative, object-orientated syntax for creating complex RF circuit and surrogate models using `JAX <https://github.com/jax-ml/jax>`_.

The library also provides tools for model optimization, fitting, statistical analysis and Bayesian inference.

:Version: |version_badge_text|
:Author: Gary Allen
:Homepage: https://github.com/gvcallen/paramrf
:Docs: https://gvcallen.github.io/paramrf
:Paper: https://doi.org/10.48550/arXiv.2510.15881

.. |tests_badge| image:: https://github.com/gvcallen/paramrf/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/gvcallen/paramrf/actions/workflows/tests.yml
   :alt: Tests Status

.. |docs_badge| image:: https://github.com/gvcallen/paramrf/actions/workflows/docs.yml/badge.svg
   :target: https://gvcallen.github.io/paramrf
   :alt: Documentation Status

.. |version_badge_text| image:: https://img.shields.io/github/v/release/gvcallen/paramrf
   :alt: GitHub Release

Features
--------

* **Declarative syntax**: Allows for the definition of models using either a self-documenting, declarative syntax, or via compositional techniques such as cascading or nodal circuit building. Also, since models can contain other models, this provides a natural means of building deeply nested, hierarchical models.
* **Differentiable**: Since the framework is built on top of ``jax`` (as opposed to ``numpy``), all models can be differentiated with respect to both frequency and parameters using *auto-differentiation*. This allows for both more efficient and flexible optimization, as well as unique design and analysis opportunities.
* **High performance and hardware flexible**: Since models are compiled using ``jax`` with Just-In-Time (JIT) compilation, model performance is improved, and models can also be computed on high-performance hardware (CPU, GPU, TPU).
* **Built-in optimization and inference wrappers**: Provides built-in wrappers for frequentist optimization and Bayesian inference in ``pmrf.optimize`` and ``pmrf.infer``, as well as high-level wrappers for fitting models to data in ``pmrf.fitting``.
* **Extensibility**: Designed to be extendable, such that additional models, fitting algorithms, cost functions, sampling routines etc. can easily be implemented.

Installation
------------
ParamRF can be installed directly using pip (requires Python 3.11+):

.. code-block:: bash

   $ pip install paramrf

Note that, for Bayesian inference or advanced statistical modeling, you may need this fork of distreqx:

.. code-block:: bash

   $ pip install git+https://github.com/gvcallen/distreqx.git

Example
-------
The example below shows how to define and optimize a simple RLC model to satisfy a given goal function. See the `documentation <https://gvcallen.github.io/paramrf>`_ for more complex examples, or have a look at the `tutorials <https://github.com/gvcallen/paramrf/tree/main/docs/tutorials>`_ folder on GitHub.

.. code-block:: python

  import pmrf as prf
  from pmrf.parameters import Scaled
  from pmrf.models import Resistor, Inductor, Capacitor
  
  freq = prf.Frequency(1, 10, 101, 'GHz')
  rlc_model = Resistor(50) ** Inductor(Scaled(1.0, 1e-9)) ** Capacitor(Scaled(1.0, 1e-12))
  
  opt_freq = prf.Frequency(4, 6, 101, 'GHz')
  goal = prf.evaluators.Goal('s11_db', '<', -20)
  
  result = prf.optimize.minimize(goal, rlc_model, opt_freq, solver=prf.optimize.ScipyMinimize())
  result.model.plot_s_db(freq, m=0, n=0)

Optional dependencies
---------------------
Several additional dependencies are required/recommended for more advanced use-cases.

For Bayesian inference, you may need this fork of distreqx:

.. code-block:: bash

   $ pip install git+https://github.com/gvcallen/distreqx

For PolyChord Bayesian inference:

.. code-block:: bash

   $ pip install git+https://github.com/PolyChord/PolyChordLite.git anesthetic mpi4py

Citation
---------------------

If you have used ParamRF for academic work, please cite the original paper (https://doi.org/10.48550/arXiv.2510.15881):
as: ::

   G.V.C. Allen, D.I.L. de Villiers, (2025). ParamRF: A JAX-native Framework for Declarative Circuit Modelling. arXiv, https://doi.org/10.48550/arXiv.2510.15881.

or using the BibTeX:

.. code:: bibtex

   @article{paramrf,
      doi = {10.48550/arXiv.2510.15881},
      url = {https://doi.org/10.48550/arXiv.2510.15881}, 
      year = {2025},
      month = {Oct},
      title = {ParamRF: A JAX-native Framework for Declarative Circuit Modelling}, 
      author = {Gary V. C. Allen and Dirk I. L. de Villiers},
      eprint = {2510.15881},
      archivePrefix = {arXiv},
      primaryClass = {cs.OH},
   }