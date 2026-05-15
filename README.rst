.. title:: Home

|tests_badge| |docs_badge|

.. image:: https://raw.githubusercontent.com/gvcallen/paramrf/main/assets/logo.png
   :align: center
   :alt: ParamRF Logo


**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modeling framework. It provides a declarative, functional syntax for creating RF circuit and surrogate models using `JAX <https://github.com/jax-ml/jax>`_.

The library provides tools for model simulation, optimization, fitting, statistical analysis, and Bayesian inference.

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

* **Declarative syntax**: Models can either be composed directly, or declared using an easy-to-read, class-based syntax.
* **Hierarchical modeling**: By nesting models within models, it is easy to create deep, hierarchical circuits.
* **Differentiable**: Since the library is built on ``jax`` (as opposed to ``numpy``), derivatives are available using *auto-differentiation*, enabling faster performance and new design opportunities.
* **Hardware flexible**: Functions are compiled just-in-time (JIT), reducing overhead and allowing computation on high-performance hardware (CPU, GPU, TPU).
* **Extensibile**: Power users can easily add additional models and algorithms by extending the library's built-in classes and interfaces.

Installation
------------
ParamRF can be installed directly using pip (requires Python 3.11+):

.. code-block:: bash

   $ pip install paramrf

Example
-------
The example below shows how to define and optimize a simple RLC model to satisfy a given goal function. See the `documentation <https://gvcallen.github.io/paramrf>`_ for more examples.

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Resistor, Inductor, Capacitor
  
  freq = prf.Frequency(1, 10, 101, 'GHz')
  rlc_model = Resistor(50) ** Inductor(prf.Value(1.0, scale=1e-9)) ** Capacitor(prf.Value(1.0, scale=1e-12))
  
  opt_freq = prf.Frequency(4, 6, 101, 'GHz')
  goal = prf.evaluators.Goal('s11_db', '<', -20)
  
  result = prf.optimize.minimize(goal, rlc_model, opt_freq, solver=prf.optimize.ScipyMinimize())
  result.model.plot_s_db(freq, m=0, n=0)

Optional dependencies
---------------------
Several additional dependencies are required/recommended for more advanced use-cases.

For Bayesian inference, you may need this fork of *distreqx*:

.. code-block:: bash

   $ pip install git+https://github.com/gvcallen/distreqx

For *BlackJAX*'s Bayesian solvers:

.. code-block:: bash

   $ pip install git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta

For the *PolyChord* solver:

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