Port Discrepancy Transfer
===========================

A port discrepancy learned from a two-port reference fit can inform a reflection-only transfer fit. The joint prediction retains the correlation between the cable parameters and its discrepancy. Attach that prediction to :class:`~pmrf.models.PortCorrected` before terminating the cable; the correction then changes the reflection seen through the load as well as the cable's transmission.

The runnable :download:`example <port_discrepancy.py>` uses a synthetic 10 m line over 10–500 MHz. The truth has skin-effect resistance and a weak frequency dependence in its inductance. The fit uses constant RLGC, with resistance parameterised by an unconstrained ``log_R`` and a docs-local :func:`~pmrf.derived` constructor. This coordinate admits a Gaussian joint prior while keeping physical resistance positive. The reference fit has a 2% prior standard deviation on log-resistance, representing a separate resistance calibration. This limits the resistance–discrepancy confounding and keeps the subsequent local Gaussian approximation useful.

Log Transmission Event Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reflection errors are additive. A cable's transmission phase winds many times over the band, so an additive transmission residual oscillates on the inverse-delay scale. A log-ratio residual removes that winding. The example defines its own bijectors, using the ``AbstractBijector`` interface available in released distreqx. For predicted transmission $S$ and observed transmission $\widetilde S$, the transform uses

$$h(\widetilde S;S)=\ln|\widetilde S| + i\left[\operatorname{unwrap}(\arg S)+\operatorname{unwrap}\left(\arg\frac{\widetilde S}{S}\right)\right].$$

Subtracting $h(S;S)$ gives the unwrapped complex log ratio. Unwrap the ratio along frequency, rather than independently unwrapping the two transmissions. Their initial ratio phase selects the local branch; transmissions must be nonzero and the frequency spacing must resolve their phase. The forward log determinant is $-2\ln|\widetilde S|$ per complex transmission entry.

.. literalinclude:: port_discrepancy.py
   :pyobject: PortEventTransform

The symmetric transmission block is the mean of the two directional logs; the antisymmetric block is their half-difference. This rotation adds the constant $-2\ln2$ to the log determinant per frequency. The full ``PortEventTransform`` is an invertible, example-local subclass of distreqx's ``AbstractBijector``. For this reciprocal model, the antisymmetric residual has no parameter sensitivity and is independent of the retained blocks in the first-order, equal-directional-noise approximation, so the fit discards it before constructing a reduced, invertible three-block transform. The discarded likelihood and rotation determinant are constants for the reference parameters.

.. literalinclude:: port_discrepancy.py
   :pyobject: ReciprocalEventTransform

Each directional transmission has noise variance $\sigma^2/\lvert S\rvert^2$ in log space to first order. Averaging independent directional measurements halves that variance. The example fixes this approximation at the observed symmetric transmission, while each additive reflection retains variance $\sigma^2$. This is a small-noise approximation, with GP hyperparameters held fixed.

Reference Fit and Joint Handoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reference likelihood uses a block-independent Matérn 5/2 GP with discrepancy amplitude 0.05 and length scale 100 MHz. Its event layout is ``(block, real_or_imaginary, frequency)``, with blocks ``('11', '22', 's21')``. The fitted additive S21 residual oscillates while the log residual changes slowly:

.. plot::
   :include-source:

   import importlib.util
   from pathlib import Path
   import matplotlib.pyplot as plt

   path = Path('port_discrepancy.py')
   spec = importlib.util.spec_from_file_location('port_discrepancy_example', path)
   example = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(example)
   result = example.run(n_frequency=150, n_transfer=30)
   frequency = result['frequency'].f_scaled
   fig, axes = plt.subplots(2, 1, sharex=True)
   axes[0].plot(frequency, result['additive_residual'].real)
   axes[0].set_ylabel('Re additive S21 residual')
   axes[1].plot(frequency, result['log_residual'][0])
   axes[1].set_ylabel('Re log-ratio residual')
   axes[1].set_xlabel('Frequency (MHz)')

:meth:`~pmrf.evaluators.MarginalLogLikelihood.predict_joint` returns a Gaussian over ``[log_R; discrepancy]``, with discrepancy flattened in that event order on the transfer grid. Its cross-covariance must survive the handoff:

.. code-block:: python

   names, joint = reference.predict_joint(fitted, frequency, transfer_frequency)
   corrected = PortCorrected(fitted, GridPortDiscrepancy(
       prf.Unconstrained(joint.mean()[1:].reshape(3, 2, n_transfer)),
       transfer_frequency, ('11', '22', 's21')))
   corrected = prf.prior(corrected, [*names, 'discrepancy.values'], joint)

The transfer fit terminates this corrected cable in a load whose resistance is a new parameter. It observes S11 only. The example optimises the cable parameters in whitened joint-prior coordinates, then linearises the transfer likelihood in declared coordinates. The dense prior costs memory quadratically and dense posterior operations cost time cubically in the number of transfer points. The plot uses a smaller transfer grid; the standalone command exercises 1000 points in both grids and reports synchronized timings:

.. code-block:: bash

   python docs/examples/port_discrepancy.py --n-frequency 1000

Quantity of Interest and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantity of interest $q$ is band-mean $|S_{21}|^2$ of the corrected, unterminated cable on the transfer grid. The fitted load informs the cable through the posterior, though $q$ has no direct load dependence. With transfer parameters $\vartheta$, :func:`~pmrf.derivative` supplies $J_q$, and :func:`~pmrf.linearization.posterior_covariance` supplies $\Sigma_\vartheta$:

$$\Sigma_q=J_q\Sigma_\vartheta J_q^\top.$$

The diagnostic recipe uses the transfer linearisation's event Jacobian $J$ and data covariance $\Sigma_D$:

$$G=\Sigma_\vartheta J^\top\Sigma_D^{-1},\qquad R=GJ,\qquad P_q=J_q(I-R),\qquad G_q=J_qG.$$

Here $\Sigma_D=\sigma^2 I$: the discrepancy is an explicit parameter with a joint prior, rather than another marginal GP in the transfer likelihood. ``P_q`` measures the QoI sensitivity left to the prior; ``G_q`` maps small reflection-data changes to QoI changes. Both are local diagnostics at the transfer MAP.

The script compares the propagated standard deviation with a nonlinear push-forward of 10,000 draws from $N(\hat\vartheta,\Sigma_\vartheta)$, evaluated in batches. The 5% standard-deviation criterion applies at amplitude 0.05; stronger log corrections can make the exponential nonlinearity too large for this linear approximation.
