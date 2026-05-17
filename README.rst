.. title:: Home

|tests_badge| |docs_badge|

.. image:: https://raw.githubusercontent.com/gvcallen/paramrf/main/assets/logo.png
   :align: center
   :alt: ParamRF Logo


**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modeling framework. It provides a declarative syntax for creating RF circuit and surrogate models using `JAX <https://github.com/jax-ml/jax>`_.

The library provides tools for model simulation, optimization, fitting, statistical analysis, and Bayesian inference.

:Version: |version_badge_text|
:Author: Gary Allen
:Repo: https://github.com/gvcallen/paramrf
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

* **Declarative syntax**: Models can be composed and defined using a self-documenting, object-oriented syntax.
* **Hierarchical modeling**: By nesting models within models, it is easy to create deep, hierarchical structures.
* **Differentiable**: Since the library is built on ``jax`` (as opposed to ``numpy``), derivatives are available via *auto-differentiation*, enabling faster performance and new design opportunities.
* **Hardware flexible**: Functions are compiled just-in-time (JIT), reducing overhead and allowing computation on high-performance hardware (CPU, GPU, TPU).
* **Extensible**: Power users can easily add additional models and algorithms by extending the library's built-in classes and interfaces.

Installation
------------
ParamRF can be installed directly using pip (requires Python 3.11+):

.. code-block:: bash

   $ pip install paramrf

Example
-------
The code below demonstrate how to define and optimize an RLC model to satisfy a given goal function. See the `documentation <https://gvcallen.github.io/paramrf>`_ for more examples.

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Resistor, Inductor, Capacitor
  
  model = Resistor(50) ** Inductor(1.0e-9) ** Capacitor(1.0e-12)
  goal = prf.evaluators.Goal('s11_db', '<', -20)
  passband = prf.Frequency(3, 4, 101, 'GHz')
  
  result = prf.optimize.minimize(goal, model, passband, solver=prf.optimize.NelderMead())
  
  plot_freq = prf.Frequency(1, 6, 101, 'GHz')
  result.model.plot_s_db(plot_freq, m=0, n=0)

Next steps
----------
* For an overview of the library's features, see the `examples <https://gvcallen.github.io/paramrf/examples/index.html>`_ page.
* For step-by-step guides that you can follow, check out the `tutorials <https://gvcallen.github.io/paramrf/tutorials/index.html>`_.
* To delve a bit deeper into the library's core building blocks and philosophy, head off to `core concepts <https://gvcallen.github.io/paramrf/core_concepts/index.html>`_.

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