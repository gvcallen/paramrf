Port Discrepancy Transfer
========================

A joint Gaussian prediction carries uncertainty in a fitted component and its port
discrepancy into a later circuit. Here a reference fit estimates a coaxial cable's
length and reciprocal port discrepancy. The transfer circuit terminates the cable
with a load and observes S11.

``predict_joint`` returns a distribution over the cable length and discrepancy.
The length is bounded, so the recipe attaches this distribution with
``truncate='unnormalised'``. This gives a valid prior for MAP fitting and posterior
covariance, but not for sampling or evidence.

The script compares the transferred posterior covariance with a NumPy Kalman
update and checks that the prior Hessian retains length-discrepancy covariance.
It evaluates the linearisation at the joint mean; it does not refit the transfer
circuit.

.. literalinclude:: port_discrepancy.py
   :pyobject: run_constrained_transfer

Run the recipe and covariance checks with:

.. code-block:: bash

   python docs/examples/port_discrepancy.py
