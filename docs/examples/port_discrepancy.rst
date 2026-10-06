Port Discrepancy Transfer
========================

A local Gaussian posterior over a reference component and its port discrepancy can
be transferred to a later circuit that observes a different quantity. This recipe
fits a constrained CoaxialLine length and a reciprocal port discrepancy, then places
the cable behind a 75-ohm load and observes its input reflection.

The reference fixture is a reciprocal coax with length 0.1 m, inner diameter
``1.12e-3`` m, outer diameter ``3.2e-3`` m, dielectric ``(ep_r=1.384,
tand=0.001)``, and conductor conductivity ``1 / 1.6e-8`` S/m. Only its length is
free, with the range ``[0.05, 0.15]`` m. The 12 reference frequencies span
100–500 MHz, and the 6 transfer frequencies span 120–480 MHz. The event blocks are
``('11', '22', 's21')``; each block stores real and imaginary parts, with frequency
last. The symmetric transmission is represented by a locally unwrapped complex
logarithm. Its discrepancy mean is reshaped in C order to ``(3, 2, 6)``.

The reference likelihood uses a Matérn 5/2 kernel with length scale 80 MHz,
discrepancy variance ``1e-6`` and jitter ``1e-12``. Each real event coordinate has
measurement variance ``1e-8``. The truth and reference observations are noise-free;
the nonzero likelihood variance remains in the covariance calculation.

``predict_joint`` returns the declared length and 36 discrepancy coordinates in one
37-dimensional Gaussian. That Gaussian has unbounded support, while the length must
remain positive. Attach it with ``truncate='unnormalised'``: the supplied Gaussian
density is unchanged inside the parameter validity, and the density is zero outside.
ParamRF does not calculate a truncation normaliser. The wrapper records
``normalised=False``; this prior supports MAP fitting, scoring, differentiation and
posterior covariance, but cannot be used for sampling or evidence. Public sampling
paths reject it before running the sampler and report the affected names.

The transfer likelihood uses S11, flattened as its real and imaginary parts in the
frequency order above, with variance ``1e-6``. The script linearises at the joint
mean and compares ParamRF's posterior covariance with an independent NumPy Kalman
update. It also checks the negative Hessian of the transferred log prior against the
inverse reference covariance, including length-discrepancy cross covariance. This
validates the supplied linearisation point; it does not perform a nonlinear transfer
refit.

The complete executable recipe is:

.. literalinclude:: port_discrepancy.py
   :pyobject: run_constrained_transfer

Run the fixture and both covariance checks with:

.. code-block:: bash

   python docs/examples/port_discrepancy.py

The reference and transfer covariance matrices are dense. Their memory cost grows
quadratically with the number of discrepancy coordinates, and dense posterior
operations grow cubically.
