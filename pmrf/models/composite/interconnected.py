"""
Composite models that physically connect ports of other models.
"""
import jax.numpy as jnp
from dataclasses import InitVar

from pmrf.models import Model, Ground
from pmrf.frequency import Frequency
from pmrf.utils import field, ArrayLike
from pmrf.models.components.ideal import Port
from pmrf.rf import cascade_a, cascade_s, terminate_s_in_s, terminate_a_in_s, renormalize_s
from pmrf.topology import Topology
from pmrf.simulate import AbstractReducer, Hallbjorner, reduce

EVAL_Z0 = 50.0

class Circuit(Model):
    """
    Represents an arbitrary interconnection of multiple `Model` objects.

    This container connects multiple models together based on a specified list
    of nodes. Each node connects one or more ports of the constituent models 
    to form a composite network.

    Parameters
    ----------
    connections : list[list[tuple[Model, int]]]
        A list representing the nodes of the circuit. Each node is a list of
        tuples, where each tuple contains a `Model` instance and the integer
        index of the port to connect to that node.
    solver : AbstractReducer, default=Hallbjorner()
        The solver to use. Available solvers can be found in :mod:`pmrf.simulate`.

    Examples
    --------
    Create a two-port PI-CLC network. External nodes are defined using `Port`, 
    and common nodes using `Ground`.

    >>> import pmrf as prf
    >>> from pmrf.models import Capacitor, Inductor, Circuit, Port, Ground
    >>> 
    >>> # Instantiate the elements, ports, and ground
    >>> C1, C2 = Capacitor(C=2e-12), Capacitor(C=1.5e-12)
    >>> L = Inductor(L=3e-9)
    >>> p0, p1, ground = Port(), Port(), Ground()
    >>> 
    >>> # Create the connections list
    >>> connections = [
    ...     [(p0, 0), (C1, 1), (L, 1)],         # Node 0 -> Port 1
    ...     [(p1, 0), (C2, 1), (L, 0)],         # Node 1 -> Port 2
    ...     [(ground, 0), (C1, 0), (C2, 0)],    # Node 2 -> Ground
    ... ]
    >>> 
    >>> # Create the circuit model
    >>> pi_clc = Circuit(connections)
    """
    
    #: The connections.
    connections: InitVar[list[list[tuple[Model, int]]]] = None
    
    #: The models in the connections.
    circuit: list[Model] = field(default=None, kw_only=True)

    #: The collated indices of the connections.
    indexed_connections: list[list[tuple[int, int]]] = field(default=None, kw_only=True, static=True)
    
    #: The circuit solver.
    solver: AbstractReducer = field(default=Hallbjorner())
    
    def __post_init__(self, connections):
        if not isinstance(connections, list):
            raise TypeError("`connections` must be a list of lists (representing nodes).")

        models = []
        indexed_connections = []
        id_to_index = {}
        seen_ports = set()

        for node_idx, connection in enumerate(connections):
            if not isinstance(connection, list):
                raise TypeError(f"Node {node_idx} in `connections` must be a list of (Model, port_index) tuples.")

            indexed_conn = []
            for item in connection:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(f"Item {item} in node {node_idx} is invalid. Must be a tuple of (Model, port_index).")

                model, value = item
                
                if not isinstance(model, Model):
                    raise TypeError(f"Expected a Model instance in node {node_idx}, got {type(model).__name__}.")
                if not isinstance(value, int):
                    raise TypeError(f"Expected an integer port index in node {node_idx}, got {type(value).__name__}.")

                model_id = id(model)
                if model_id not in id_to_index:
                    id_to_index[model_id] = len(models)
                    models.append(model)
                
                model_idx = id_to_index[model_id]
                
                if value < 0 or value >= model.nports:
                    raise ValueError(f"Port index {value} out of bounds for model of type {type(model)} (name = '{getattr(model, 'name', 'unnamed')}', nports={model.nports}).")
                
                # Prevent the same port of the same model instance from being connected to multiple nodes
                port_signature = (model_id, value)
                if port_signature in seen_ports:
                    raise ValueError(f"Port {value} of model named '{getattr(model, 'name', 'unnamed')}' is connected multiple times. A port can only belong to one node.")
                seen_ports.add(port_signature)

                indexed_conn.append((model_idx, value))
            
            indexed_connections.append(indexed_conn)
            
        # Assign the computed values
        self.circuit = models
        self.indexed_connections = indexed_connections
        
    @property
    def topology(self) -> Topology:
        port_idx = []
        ground_idxs = []
        offset = 0
        
        for model in self.circuit:
            if isinstance(model, Port):
                for p in range(model.nports):
                    port_idx.append(offset + p)
            elif isinstance(model, Ground):
                for p in range(model.nports):
                    ground_idxs.append(offset + p)
            offset += model.nports        
        
        return Topology(self.circuit, self.indexed_connections, port_idx, ground_idxs)
        
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        result = reduce(self.topology, freq, self.solver, z0=z0)
        return result.s
    
    def y(self, freq: Frequency) -> jnp.ndarray:
        result = reduce(self.topology, freq, self.solver)
        return result.y
    

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

    Parameters
    ----------
    cascade : tuple[Model]
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
    #: The models.
    cascade: tuple[Model]
    
    def __post_init__(self):
        for model in self.cascade:
            if model.nports % 2 != 0:
                raise ValueError('All networks must be 2N-ports for Cascade')
            
    @property
    def merged_cascade(self) -> list[Model]:
        """
        Returns the models of the cascade merged such that any cascades are combined.

        This is done only during the forward pass to retain the caller's
        original nesting for debugging/inspection purposes.
        """
        merged = []
        for model in self.cascade:
            if isinstance(model, Cascade):
                merged.extend(model.merged_cascade)
            else:
                merged.append(model)
        return merged
    
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0):
        merged_models = self.merged_cascade
        
        Smats = jnp.array([model.s(frequency, z0=EVAL_Z0) for model in merged_models])
        z0s = jnp.array([EVAL_Z0 for _ in merged_models])
        
        Scas, _ = cascade_s(Smats, z0s)
        return renormalize_s(Scas, z_old=EVAL_Z0, z_new=z0, s_def_old='power', s_def_new='power')
    
    def a(self, frequency: Frequency):
        merged_models = self.merged_cascade
        Amats = jnp.array([model.a(frequency) for model in merged_models])
        return cascade_a(Amats)
    
    
class Terminated(Model):
    """
    Represents one network terminated in another.

    Parameters
    ----------

    terminated_from : Model
        The model being terminated.
    terminated_into : Model
        The model that `terminated_from` is terminated into.
    """
    #: The "from" model.
    terminated_from: Model

    #: The "into" model.
    terminated_into: Model
    
    def __post_init__(self):
        if self.terminated_from.nports != 2*self.terminated_into.nports:
            raise ValueError("Terminated only supports terminating 2N port networks in a 1N port")
        
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        Smat_from = self.terminated_from.s(frequency, z0=EVAL_Z0)
        Smat_into = self.terminated_into.s(frequency, z0=EVAL_Z0)
        
        S_term, _ = terminate_s_in_s(Smat_from=Smat_from, z0_from=EVAL_Z0, Smat_into=Smat_into, z0_into=EVAL_Z0)
        return renormalize_s(S_term, EVAL_Z0, z0, 'power', 'power')