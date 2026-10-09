scikit-rf Comparison
====================

ParamRF aims to complement and not replace scikit-rf. While scikit-rf excels at static network manipulation, it does not take a parameter-first modeling approach, and also suffers from Python interpreter overhead and finite difference approximations which JAX eleviates.

This section provides a brief comparison between the core philosophies of scikit-rf and ParamRF, and also demonstrates a short benchmarking script showing the performance benefits when using JAX.

.. toctree::
   :maxdepth: 2

   overview
   performance