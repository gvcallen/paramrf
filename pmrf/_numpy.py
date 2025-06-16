import os

# use_jax = os.environ.get("JAX_ENABLED") == "1"
USE_JAX = True

if USE_JAX:
    import jax.numpy as numpy
else:
    import numpy as nump