from typing import Sequence

import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.models.misc import Port
from pmrf.functions.connections import connect_one, connect_many
from pmrf._util import field

class Circuit(Model):
    """
    Represents an arbitrary circuit defined by component connections.

    This model allows for the definition of a circuit by specifying how the ports
    of various sub-models are connected together.

    Attributes
    ----------
    models : list[Model]
        The list of unique models involved in the circuit.
    indexed_connections : list[list[tuple[int, int]]]
        Internal representation of connections using model indices instead of objects.
    port_idxs : list[int]
        Indices of the models that act as external ports for the circuit.
    """
    models: list[Model]
    indexed_connections: list[list[tuple[int, int]]] = field(static=True)
    port_idxs: list[int] = field(static=True)

    def __init__(self, connections: list[list[tuple[Model, int]]]):
        """
        Initialize the Circuit.

        Parameters
        ----------
        connections : list[list[tuple[Model, int]]]
            A list of connections (nodes). Each connection is a list of
            `(model_instance, port_index)` tuples that are electrically connected.
        """
        super().__init__()

        self.models = []
        self.indexed_connections = []
        self.port_idxs = []
        id_to_index: dict[Model, int] = {}

        for connection in connections:
            indexed_connection = []
            for model, value in connection:
                if id(model) not in id_to_index:
                    id_to_index[id(model)] = len(self.models)
                    self.models.append(model)
                model_idx = id_to_index[id(model)]
                indexed_connection.append((model_idx, value))
            self.indexed_connections.append(indexed_connection)
        for model in self.models:
            if isinstance(model, Port):
                self.port_idxs.append(id_to_index[id(model)])

    def s(self, freq: Frequency) -> jnp.array:
        Smats = [model.s(freq) for model in self.models]
        z0s = [model.z0 for model in self.models]

        return connect_many(Smats, z0s, self.indexed_connections, self.port_idxs)

class Connected(Model):
    """
    Represents a connection of multiple models at a single intersection.

    The algorithm in :func:`pmrf.functions.connections.connect_one` is used.

    Attributes
    ----------
    models : Sequence[Model] | Model
        The models to connect.
    ports : Sequence[int | Sequence[int]]
        The port indices on each model that are connected to the common node.
    """    
    models: Sequence[Model] | Model
    ports: Sequence[int | Sequence[int]]

    def __post_init__(self):
        self.name = 'connected'
        # Do some validation

    def s(self, freq: Frequency) -> jnp.array:
        models, ports = self.models, self.ports
        if isinstance(models, Model):
            models = [models]

        Smats = [model.s(freq) for model in models]
        z0s = [model.z0 for model in models]
        Sout, _ = connect_one(Smats, z0s, ports)
        return Sout

class Cascade(Model):
    """
    Represents a cascade, or series connection, of two or more `Model` objects.

    This container connects multiple models end-to-end. The output port of
    one model is connected to the input port of the next. This is mathematically
    equivalent to chain-multiplying the ABCD-parameter matrices of the
    constituent models.

    The `Cascade` model automatically flattens any nested `Cascade` instances
    to maintain a simple, linear chain of models. The number of ports of the
    resulting `Cascade` network depends on the port count of the final model
    in the chain.

    Attributes
    ----------
    models : tuple[Model]
        The sequence of models in the cascade.

    Examples
    --------
    Cascading models is most easily done using the `**` operator, which is
    an alias for creating a `Cascade` model.

    >>> import pmrf as prf
    >>> from pmrf.models import Resistor, Capacitor, Inductor

    # Create individual component models
    >>> res = Resistor(50)
    >>> cap = Capacitor(1e-12)
    >>> ind = Inductor(1e-9)

    # Cascade them together in a series R-L-C configuration
    # This is equivalent to Cascade(models=(res, ind, cap))
    >>> rlc_series = res ** ind ** cap

    # Define a frequency axis
    >>> freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')

    # Calculate the S-parameters of the cascaded network
    >>> s_params = rlc_series.s(freq)

    >>> print(f"Cascaded model has {rlc_series.nports} ports.")
    >>> print(f"S11 at first frequency point: {s_params[0,0,0]:.2f}")
    """

    models: tuple[Model]
    
    def __post_init__(self):
        self.name = 'cascade'
        
        models = self.models
        # First check all the port conditions
        if models[0].nports != 2:
            raise Exception('First network must be a two port when cascade')
        for model in models[1:-1]:
            if model.nports != 2:
                raise Exception('Inner networks must be two ports when cascade')
        if models[-1].nports not in (1, 2):
            raise Exception('Last network must either be a one port or a two port when cascade')
        
        # Next check if any models themselves are of type Cascade. We don't nest these - we chain them to avoid very deep, nested models
        model_reduced = []
        for model in models:
            if isinstance(model, Cascade):
                model_reduced.extend(model.models)
            else:
                model_reduced.append(model)

        self.models = tuple(model_reduced)

    @property
    def first_model(self) -> Model:
        """Model: The first model in the cascade chain."""
        return self.models[0]
    
    @property
    def inner_models(self) -> tuple['Model']:
        """tuple[Model]: A tuple of the inner models in the cascade chain."""
        return tuple(self.models[1:-1])
    
    @property
    def last_model(self) -> Model:
        """Model: The last model in the cascade chain."""
        return self.models[-1]

    def a(self, freq: Frequency) -> jnp.ndarray:
        a = self.first_model.a(freq)
        for model in self.models[1:]:
            a = a @ model.a(freq)
        if self.last_model.nports == 1:
            raise Exception('Cannot get abcd-matrix for a cascade of models terminated in a one-port')
        
        return a

    def s(self, freq: Frequency) -> jnp.ndarray:
        # We only implement s when we are terminating in a one-port.
        # Otherwise, we call the parent s, which will ultimatlely call the 'a' implementation above
        if self.last_model.nports != 1:
            return Model.s(self, freq)
        
        # Get abcd matrix of inners
        a = self.first_model.a(freq)
        for model in self.inner_models:
            a = a @ model.a(freq)
        
        # Terminated last in s11
        s11 = self.last_model.s(freq)[:,0,0]
        z0 = self.first_model._z0
        
        A, B, C, D = a[:,0,0], a[:,0,1], a[:,1,0], a[:,1,1]
        num = z0 * (1 + s11) * (A - z0*C) + (B - D*z0)*(1-s11)
        den = z0 * (1 + s11) * (A + z0*C) + (B + D*z0)*(1-s11)
        s11_out = num / den        
        return s11_out.reshape(-1, 1, 1) 
    
class Renumbered(Model):
    """
    A container that re-numbers the ports of a given `Model`.

    This is useful for creating complex network topologies by explicitly
    re-mapping the port indices of a sub-network.
    
    Attributes
    ----------
    model : Model
        The underlying model to renumber.
    to_ports : tuple[int]
        The new port indices.
    from_ports : tuple[int]
        The original port indices that map to `to_ports`.
    """
    model: Model
    to_ports: tuple[int]
    from_ports: tuple[int]

    def __post_init__(self):
        self.name = 'renumbered'
        
        model = self.model
        to_ports, from_ports = self.to_ports, self.from_ports

        if model.primary_property == 'a' and len(from_ports) != 2 and len(to_ports) != 2:
            raise ValueError("(from_ports, to_ports) must be either (0, 1) or (1, 0) for 'a' primary networks")        

    def renumber(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the port renumbering to a parameter matrix.

        Parameters
        ----------
        p : jnp.ndarray
            The parameter matrix to renumber (e.g., S-parameters).

        Returns
        -------
        jnp.ndarray
            The renumbered parameter matrix.
        """
        p_new = p.copy()
        p_new = p_new.at[:, self.to_ports, :].set(p[:, self.from_ports, :])
        p_new = p_new.at[:, :, self.to_ports].set(p_new[:, :, self.from_ports])
        return p_new
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.a(freq))

    def s(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.s(freq))
    
class Flipped(Renumbered):
    """
    A model container that flips the ports of a multi-port network.

    For a 2-port network, this is equivalent to swapping port 1 and port 2.
    For a 4-port network, ports (1,2) are swapped with (3,4), and so on.
    This is a convenient specialization of the `Renumbered` model.
    """
    to_ports: tuple[int] = field(init=False)
    from_ports: tuple[int] = field(init=False)

    def __post_init__(self):
        if self.model.nports % 2 != 0:
            raise ValueError("You can only flip multiple-of-two-port Networks")
        
        n = int(self.model.nports / 2)
        self.to_ports = tuple(range(0, 2 * n))
        self.from_ports = tuple(range(n, 2 * n)) + tuple(range(0, n))

        super().__post_init__()

        self.name = 'flipped'
        
class Stacked(Model):
    """
    A container that stacks multiple models in a block-diagonal fashion.

    This combines several `Model` objects into a single, larger model where
    the individual S-parameter matrices are placed along the diagonal of the
    combined S-parameter matrix. This represents a set of unconnected
    networks treated as a single component.

    Attributes
    ----------
    models : tuple[Model, ...]
        The models to stack.
    """
    models: tuple[Model, ...]
    
    def __post_init__(self):
        self.name = 'stacked'
        
    def s(self, freq: Frequency) -> jnp.ndarray:
        num_ports = sum(model.nports for model in self.models)

        s = jnp.zeros((freq.npoints, num_ports, num_ports), dtype=jnp.complex128)
        i = 0
        for submodel in self.models:
            s_sub = submodel.s(freq)
            n_sub = submodel.nports
            
            s = s.at[:,i:i+n_sub,i:i+n_sub].set(s_sub)
            
            i += n_sub
        return s