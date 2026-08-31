Sharing a Substrate Between Traces
==================================

A board has one substrate and many traces on it. If each trace carried its own
copy of the permittivity, fitting the board would fit the same physical quantity
several times over, once per trace, with nothing keeping the answers consistent.

ParamRF needs no special machinery for this. A :class:`pmrf.materials.Substrate`
groups the height, dielectric, conductor and metallization thickness into one
module, and a :class:`pmrf.models.AbstractBuilder` holding one substrate and
injecting it into each line in ``build`` is what makes the sharing real.

One Substrate, Two Traces
~~~~~~~~~~~~~~~~~~~~~~~~~

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
   list(board.named_params())

There is one ``ep_r``, not two. ``build`` is lazy and uncached, so the substrate
is a leaf of the builder and both lines are reconstructed on every call from the
same already-traced values. Fitting ``board`` therefore fits a single
permittivity, shared by construction.

The Contrast: A Shared ``Param`` Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handing the *same* :class:`pmrf.Param` object to two separately-constructed
lines looks like sharing, but is not:

.. jupyter-execute::

   ep_r = prf.Param(value=4.3)
   lines = (
       MicrostripLine(w=1e-3, h=1.6e-3, dielectric=ConstantDielectric(ep_r=ep_r), length=0.1)
       ** MicrostripLine(w=2e-3, h=1.6e-3, dielectric=ConstantDielectric(ep_r=ep_r), length=0.2)
   )
   [name for name in lines.named_params() if name.endswith("ep_r")]

PyTree flattening gives one leaf per line regardless of the object identity that
went in, so the optimizer sees two independent permittivities that are free to
drift apart. The builder pattern above is the one that dedupes.

Two Idioms, One PyTree
~~~~~~~~~~~~~~~~~~~~~~

A line may be given a grouped ``substrate=``, or the loose ``h``,
``dielectric``, ``conductor`` and ``t`` fields. Both build the same canonical
:class:`pmrf.materials.Substrate`, so the two forms are the same model:

.. jupyter-execute::

   import jax

   loose = MicrostripLine(w=3e-3, h=1.6e-3, dielectric=4.3, length=0.1)
   grouped = MicrostripLine(w=3e-3, substrate=Substrate(h=1.6e-3, dielectric=4.3), length=0.1)

   jax.tree_util.tree_structure(loose) == jax.tree_util.tree_structure(grouped)

The loose form stays convenient for a single line; the grouped form is what a
board shares. They may not be mixed in one call.

A coaxial line has no substrate, and keeps its ``dielectric`` and ``conductor``
fields directly.
