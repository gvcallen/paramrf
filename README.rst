|tests_badge| |docs_badge|

.. image:: https://raw.githubusercontent.com/gvcallen/paramrf/main/assets/logo.png
   :align: center
   :alt: ParamRF Logo


**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modelling framework. It provides a declarative, object-orientated syntax for modelling complex RF circuits and surrogates using `JAX <https://github.com/jax-ml/jax>`_ and `Equinox <https://github.com/patrick-kidger/equinox>`_. The library also provides tools for model optimization, fitting, statistical analysis and Bayesian inference.

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

Key Features
---------------------

* **Declarative syntax**: Allows for the definition of models using either a self-documenting, declarative syntax, or via compositional techniques such as cascading or node composition. Since models can consist of a mix of `parax.Parameter <https://gvcallen.github.io/parax/api/#parax.Parameter>`_ and other ``pmrf.Model`` objects, this allows for a natural means of building complex, hierarchial models.
* **Differentiable**: Since the framework is built using ``jax``, all models can be differentiated with respect to frequency and parameters. This allows for complex optimization and sensitivity analysis.
* **High performance and hardware flexibile**: Since models are compiled using ``jax`` with Just-In-Time (JIT) compilation, model performance is improved, and models can also be computed on high-performance hardware (CPU, GPU, TPU).
* **Built-in optimization and inference wrappers**: Provides built-in wrappers for frequentist optimization and Bayesian inference in ``pmrf.optimize`` and ``pmrf.infer``, as well as high-level wrappers for data-fitting in ``pmrf.fit``.
* **Extensibility**: Designed to be extendable, such that additional models, fitting algorithms, cost functions, sampling routines etc. can easily be implemented.

Example
---------------------
The example below shows how to define and fit a simple RLC model to measured data using ParamRF. See the `documentation <https://gvcallen.github.io/paramrf>`_ for more complex examples.

.. code:: python

    import skrf as rf
    import pmrf as prf
    from pmrf.models import Resistor, Inductor, Capacitor
    from pmrf.fit import fit

    # Define the model and load data
    model = Resistor(R=100.0) ** Inductor(L=1e-9) ** Capacitor(L=1e-12)
    data = rf.Network('path/to/rlc.s2p')
    
    # Fit the model and output results and parmaeters
    results = fit(model, data)
    results.plot('s_db')
    print(results.model.named_params())

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