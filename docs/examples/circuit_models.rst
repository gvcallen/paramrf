Circuit Models
==============

For arbitrarily complex circuits, ParamRF offers the ability to combine models using the :class:`~pmrf.models.composite.interconnected.Circuit` class. For those familiar, the syntax is the same as scikit-rf's :class:`skrf.circuit.Circuit` class.

Overview
~~~~~~~~

:class:`~pmrf.models.composite.interconnected.Circuit` accepts a list of "connections". Each entry in this list is a node in the circuit. Each node is another list, with each element being a tuple for each connected circuit element or sub-model. Each tuple then contains the model object, as well as the index of the port for that model that is connected in that node. If this sounds confusing, don't worry - the method is easy to understand using a simple example.

Defining the Connections
~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider a two-port PI-CLC network as illustrated below. "External" nodes (each entry in the outer list) are numbered as E0, E1 etc. whereas "internal" port indices (ports for each model in the circuit) are numbered per element as I0, I1 etc.

.. image:: circuit_clc.png
   :alt: pi-CLC circuit diagram
   :width: 600px
   :align: center

First, let's create the components and set up the connections list:

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Capacitor, Inductor, Port, Ground
  
  C1, C2, L = Capacitor(C=2e-12), Capacitor(C=1.5e-12), Inductor(L=3e-9)
  p0, p1, ground = Port(), Port(), Ground()
  
  connections = [
      [(p0, 0), (C1, 1), (L, 1)],         # E0 -> port 1
      [(p1, 0), (C2, 1), (L, 0)],         # E1 -> port 2
      [(ground, 0), (C1, 0), (C2, 0)],    # E2
  ]

Note that the ports will be exposed in our new model in the order they are defined in the list.

Creating the Circuit
~~~~~~~~~~~~~~~~~~~~

Finally, we can feed our connection definition into :class:`~pmrf.models.composite.interconnected.Circuit`:

.. code-block:: python

  from pmrf.models import Circuit
  pi_clc = Circuit(connections)

``pi_clc`` is a new model of type :class:`~pmrf.models.composite.interconnected.Circuit`. We can easily plot its S-parameters:

.. code-block:: python
  
  freq = prf.Frequency(1, 1000, 1001, 'MHz')
  pi_clc.plot_s_db(freq, m=1, n=0)

Note that the ``m`` and ``n`` indices represent the ports (zero-based).