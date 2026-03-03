Installation
=====================
ParamRF can be installed using pip directly from the GitHub page:

``pip install git+https://github.com/paramrf/paramrf@main``

Optional dependencies
---------------------
Several additional dependencies are required/recommended for more advanced use-cases.

For PolyChord fitting:

``
pip install git+https://github.com/PolyChord/PolyChordLite.git anesthetic mpi4py
``

For BlackJAX fitting:

``
pip install git+https://github.com/handley-lab/blackjax@nested_sampling anesthetic
``

For eqx-learn surrogate modeling:

``
pip install git+https://github.com/eqx-learn/eqx-learn
``