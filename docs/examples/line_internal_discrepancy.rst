Line internal discrepancy across lengths
========================================

A port correction carries the error of a particular line instance. A correction
to characteristic impedance, attenuation and phase constant can be carried to a
different length when the nominal line's electrical length scales with its
physical length. The line must be lossy, and the frequency grid must have
positive forward phase.

This example uses a 30 cm lossy coax model with 10% excess conductor loss. It
transfers its correction to 15 cm. The comparison reports RMS S21 error and the
band-mean bias of ``eta = |S21|² / (1 - |S11|²)``.
Download the :download:`complete executable script <line_internal_discrepancy.py>`.

.. literalinclude:: line_internal_discrepancy.py
   :language: python
   :start-at: def compare_transfer
   :end-before: def reference_route

The reference route transforms measured ``(Zc, gamma)`` into the carrier's log
features. Noise variance belongs to that transformed event space. Its joint
prediction includes the free conductor parameter and attaches by name to the
shorter line.

.. literalinclude:: line_internal_discrepancy.py
   :language: python
   :start-at: def reference_route
   :end-before: def _basis

When only S is observed, the second route fits standard-normal basis
coefficients through complex reflection and transmission. The kernel amplitude
and lengthscale are fixed inputs. A Laplace covariance supplies ``mean`` and
``sigma`` arrays for comparison with the injected correction; the recipe reports
the fraction lying within ``mean ± 2σ``.

.. literalinclude:: line_internal_discrepancy.py
   :language: python
   :start-at: def _basis
   :end-before: if __name__

Run the three recipes and inspect the four correction channels in block order
``(log Zc real, log Zc imaginary, log alpha, log beta)``:

.. code-block:: python

   comparison = compare_transfer()
   reference = reference_route()
   s_only = s_only_route()
   lower = s_only['mean'] - 2 * s_only['sigma']
   upper = s_only['mean'] + 2 * s_only['sigma']
   coverage = s_only['coverage']
