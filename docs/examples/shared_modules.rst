Sharing a Module Across Models
==============================

A model normally owns its modules and their parameters. To share one module across
several models, store it on an :class:`pmrf.models.AbstractBuilder` and pass it to
each model in ``build``. This board shares one substrate between two traces.


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
