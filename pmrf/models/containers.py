from typing import Sequence

import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.models.ideal import Port
from pmrf.rf_functions.connections import connect_s_arbitrary, cascade_a, cascade_s, terminate_s_in_s
from pmrf._util import field

class Container(Model):
    pass

class Circuit(Container):
    """
    Represents an arbitrary circuit defined by component connections.

    This model allows for the definition of a circuit by specifying how the ports
    of various sub-models are connected together.

    NB: The ports numbers are exposed in the order they appear in the connections list.

    Attributes
    ----------
    models : list[Model]
        The list of unique models involved in the circuit.
    indexed_connections : list[list[tuple[int, int]]]
        Internal representation of connections using model indices instead of objects.
        Ports are exposed in the order they appear in the list.
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
                if value > model.nports - 1:
                    raise Exception(f"Port index out of bounds for model {model} in Circuit")
            self.indexed_connections.append(indexed_connection)
        for model in self.models:
            if isinstance(model, Port):
                self.port_idxs.append(id_to_index[id(model)])

    def s(self, freq: Frequency) -> jnp.array:
        Smats = [model.s(freq) for model in self.models]
        z0s = [model.z0 for model in self.models]

        Scon, _z0con = connect_s_arbitrary(Smats, z0s, self.indexed_connections, self.port_idxs)
        return Scon

class Cascade(Container):
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
        model_reduced = []
        for model in models:
            # Check that ports are a multiple of 2
            if model.nports % 2 != 0:
                raise Exception('All networks must be 2N-ports for Cascade')
            if isinstance(model, Cascade):
                model_reduced.extend(model.models)
            else:
                model_reduced.append(model)
        self.models = tuple(model_reduced)

    def a(self, freq: Frequency) -> jnp.ndarray:
        return cascade_a([model.a(freq) for model in self.models])

    def s(self, freq: Frequency) -> jnp.ndarray:
        Smats = jnp.array([model.s(freq) for model in self.models])
        z0s = jnp.array([model.z0 for model in self.models])
        Scas, z0cas = cascade_s(Smats, z0s)
        return Scas
    
class Terminated(Container):
    """
    Represents one network terminated in another.

    Currently, this only supports terminating a two-port network in a one-port.
    
    Attributes
    ----------
    model_from: Model
        The model to terminate from.
    model_into: Model
        The model to terminate into.
    """
    from_model: Model
    into_model: Model
    
    def __post_init__(self):
        self.name = 'terminated'

        if self.from_model.nports != 2 or self.into_model.nports != 1:
            raise Exception("Currently, Terminated only supports 2-port networks terminated in a 1-port")

    def s(self, freq: Frequency) -> jnp.ndarray:
        Smat_from = self.from_model.s(freq)
        z0_from = self.from_model.z0
        Smat_into = self.into_model.s(freq)
        z0_into = self.into_model.z0
        S_term, z0_term = terminate_s_in_s(Smat_from, z0_from, Smat_into, z0_into)
        return S_term
        
class Renumbered(Container):
    """
    A container that re-numbers the ports of a given `Model`.

    This is useful for creating complex network topologies by explicitly
    re-mapping the port indices of a sub-network.
    
    Attributes
    ----------
    model : Model
        The underlying model to renumber.
    from_ports : tuple[int]
        The original port indices that map to `to_ports`.
    to_ports : tuple[int]
        The new port indices. Can be `None`, in which case `from_ports`
        must contain exactly two ports to be swapped.
    """
    model: Model
    from_ports: tuple[int]
    to_ports: tuple[int] = None

    def __post_init__(self):
        self.name = 'renumbered'
        
        model = self.model
        if self.to_ports is None:
            if len(self.from_ports) != 2:
                raise Exception("from_ports must have length==2 if to_ports is None")
            self.to_ports = (self.from_ports[1], self.from_ports[0])
        
        if model.primary_property == 'a' and len(self.from_ports) != 2 and len(self.to_ports) != 2:
            raise ValueError("(from_ports, to_ports) must be either (0, 1) or (1, 0) for 'a' primary networks")        
        
        if len(self.from_ports) != len(self.to_ports):
            raise ValueError("from_ports and to_ports must have the same length for Renumbered")

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

    def y(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.y(freq))

    def z(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.z(freq))
    
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
        
class Stacked(Container):
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