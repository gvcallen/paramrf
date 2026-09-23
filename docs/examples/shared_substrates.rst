Sharing a Substrate Between Traces
==================================

A board has one substrate and many traces. However, because of ParamRF's design, each transmission line has its own set of parameters (by default).

To share parameters across lines, for example sharing a substrate, create a single :class:`pmrf.materials.Substrate`, store it in a
:class:`pmrf.models.AbstractBuilder`, and pass it to each line in ``build``.

.. jupyter-execute::

   import pmrf as prf
   from pmrf.materials import ConstantDielectric, Substrate
   from pmrf.models import AbstractBuilder, MicrostripLine

   class Board(AbstractBuilder):
       substrate: Substrate
       w1: prf.Param
       w2: prf.Param

       def build(self):
           return (
               MicrostripLine(w=self.w1, substrate=self.substrate, length=0.1)
               ** MicrostripLine(w=self.w2, substrate=self.substrate, length=0.2)
           )

   board = Board(
       substrate=Substrate(h=1.6e-3, dielectric=ConstantDielectric(ep_r=4.3, tand=0.02)),
       w1=1.0e-3,
       w2=2.0e-3,
   )
   list(prf.params(board))

Grouped and Loose Fields
~~~~~~~~~~~~~~~~~~~~~~~~

A line accepts either a grouped ``substrate=`` or the loose ``h``,
``dielectric``, ``conductor`` and ``t`` fields. Both produce the same
:class:`pmrf.materials.Substrate`, so the two forms give the same model:

.. jupyter-execute::

   import jax

   loose = MicrostripLine(w=3e-3, h=1.6e-3, dielectric=4.3, length=0.1)
   grouped = MicrostripLine(w=3e-3, substrate=Substrate(h=1.6e-3, dielectric=4.3), length=0.1)

   jax.tree_util.tree_structure(loose) == jax.tree_util.tree_structure(grouped)

The loose form is convenient for a single line, and the grouped form lets a board
pass one object to every line.