"""
Composite models that physically connect ports of other models.
"""
import jax.numpy as jnp
from dataclasses import InitVar

from pmrf.models import Model, Ground
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.types import ArrayLike
from pmrf.models.components.ideal import Port
from pmrf.simulate.topology import Topology
from pmrf.simulate import (
    AbstractReducer, 
    AbstractCascader, 
    AbstractTerminator, 
    GlobalSchurScatteringReducer, 
    SequentialSchurScatteringReducer,    
    AnalyticScatteringTerminator, 
    BlockSchurScatteringReducer,
    RedhefferScatteringCascader, 
    reduce, 
    cascade, 
    terminate
)

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
    solver : AbstractReducer, default=HallbjornerReducer()
        The solver to use. Available solvers can be found in :mod:`pmrf.simulate`.
    flatten: bool, default=True
        (experimental) Flattens the connections into one large circuit if they contain sub-circuits.
        Can improve performance for small circuits, but may reduce performance for large circuits.

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
    
    #: Flattens the connections.
    flatten: InitVar[bool] = field(default=True, static=True, kw_only=True)
    
    #: The models in the connections.
    circuit: list[Model] = field(default=None, kw_only=True)

    #: The collated indices of the connections.
    indexed_connections: list[list[tuple[int, int]]] = field(default=None, kw_only=True, static=True)
    
    #: The circuit solver.
    solver: AbstractReducer = field(default=GlobalSchurScatteringReducer())
    
    @staticmethod
    def _flatten_connections(connections):
        """Flattens nested Circuit instances into base models using ID-based set operations."""
        model_map = {}
        
        # Initialize nodes with integer IDs instead of instances to bypass JAX hashing limitations
        nodes = []
        for connection in connections:
            node_set = set()
            for model, port in connection:
                model_map[id(model)] = model
                node_set.add((id(model), port))
            nodes.append(node_set)
            
        while True:
            # Find a nested Circuit
            sub_c_id = None
            sub_c = None
            for node in nodes:
                for mid, _ in node:
                    if isinstance(model_map[mid], Circuit):
                        sub_c_id = mid
                        sub_c = model_map[mid]
                        break
                if sub_c_id: break
                
            if not sub_c_id:
                break # Hierarchy is completely flat
                
            # Extract internal layout of the sub-circuit
            sub_ports = [m for m in sub_c.circuit if isinstance(m, Port)]
            for p in sub_ports:
                model_map[id(p)] = p
                
            # Replace the parent's reference to the sub-circuit with the internal Port ID
            for node in nodes:
                items_to_remove, items_to_add = [], []
                for item in node:
                    mid, port_idx = item
                    if mid == sub_c_id:
                        items_to_remove.append(item)
                        items_to_add.append((id(sub_ports[port_idx]), 0))
                
                for item in items_to_remove: node.remove(item)
                for item in items_to_add: node.add(item)
                    
            # Bring in all internal nodes from the sub-circuit
            for idx_conn in sub_c.indexed_connections:
                internal_node = set()
                for m_idx, p_idx in idx_conn:
                    m = sub_c.circuit[m_idx]
                    model_map[id(m)] = m
                    internal_node.add((id(m), p_idx))
                nodes.append(internal_node)
                
            # Merge any intersecting nodes
            merged_nodes = []
            for node in nodes:
                if not node: continue
                intersecting = [m for m in merged_nodes if m.intersection(node)]
                if intersecting:
                    new_set = set.union(node, *intersecting)
                    merged_nodes = [m for m in merged_nodes if m not in intersecting]
                    merged_nodes.append(new_set)
                else:
                    merged_nodes.append(node)
            nodes = merged_nodes
            
            # Drop the sub-circuit's dummy Ports entirely
            sub_port_ids = {id(p) for p in sub_ports}
            for node in nodes:
                # Find elements whose model ID belongs to a dummy port
                to_discard = [item for item in node if item[0] in sub_port_ids]
                for item in to_discard:
                    node.discard(item)
                    
            # Clean up empty sets
            nodes = [n for n in nodes if n]

        # Convert the sets back into lists, resolving the IDs back to the actual Model instances
        return [[(model_map[mid], port) for mid, port in n] for n in nodes]

    def __post_init__(self, connections: list, flatten: bool):
        # Input validation
        if not isinstance(connections, list):
            raise TypeError("`connections` must be a list of lists (representing nodes).")

        seen_ports = set()
        for node_idx, connection in enumerate(connections):
            if not isinstance(connection, list):
                raise TypeError(f"Node {node_idx} in `connections` must be a list of (Model, port_index) tuples.")

            for item in connection:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(f"Item {item} in node {node_idx} is invalid. Must be a tuple of (Model, port_index).")

                model, value = item
                
                if not isinstance(model, Model):
                    raise TypeError(f"Expected a Model instance in node {node_idx}, got {type(model).__name__}.")
                if not isinstance(value, int):
                    raise TypeError(f"Expected an integer port index in node {node_idx}, got {type(value).__name__}.")

                if value < 0 or value >= model.nports:
                    raise ValueError(f"Port index {value} out of bounds for model of type {type(model)} (name = '{getattr(model, 'name', 'unnamed')}', nports={model.nports}).")
                
                port_signature = (id(model), value)
                if port_signature in seen_ports:
                    raise ValueError(f"Port {value} of model named '{getattr(model, 'name', 'unnamed')}' is connected multiple times. A port can only belong to one node.")
                seen_ports.add(port_signature)

        if flatten:
            connections = self._flatten_connections(connections)

        models = []
        indexed_connections = []
        id_to_index = {}
        seen_ports = set()

        for node_idx, connection in enumerate(connections):
            indexed_conn = []
            for item in connection:
                model, value = item
                model_id = id(model)
                if model_id not in id_to_index:
                    id_to_index[model_id] = len(models)
                    models.append(model)
                
                model_idx = id_to_index[model_id]
                
                # Prevent the same port of the same model instance from being connected to multiple nodes
                port_signature = (model_id, value)
                seen_ports.add(port_signature)

                indexed_conn.append((model_idx, value))
            
            indexed_connections.append(indexed_conn)
            
        # Assign the computed values
        self.circuit = models
        self.indexed_connections = indexed_connections

    @property
    def number_of_ports(self):
        i = 0
        for model in self.circuit:
            if isinstance(model, Port):
                i = i + 1
        return i
        
    @property
    def topology(self) -> Topology:
        port_indices = []
        ground_indices = []
        offset = 0
        
        for model in self.circuit:
            if isinstance(model, Port):
                for p in range(model.nports):
                    port_indices.append(offset + p)
            elif isinstance(model, Ground):
                for p in range(model.nports):
                    ground_indices.append(offset + p)
            offset += model.nports
        
        return Topology(
            components=self.circuit,
            indexed_connections=self.indexed_connections,
            port_indices=port_indices,
            ground_indices=ground_indices,
            marker_indices=port_indices,
        )
        
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return reduce(self.topology, freq, self.solver, z0=z0).s
    
    def y(self, freq: Frequency) -> jnp.ndarray:
        return reduce(self.topology, freq, self.solver).y
    

class Parallel(Model):
    """
    Represents a parallel connection of two or more `Model` objects.

    This container connects multiple models in parallel. Port `i` of every 
    model is connected to port `i` of every other model, forming a single 
    composite network with the same number of ports as the individual models.

    All models must have the exact same number of ports.

    Parameters
    ----------
    parallel : tuple[Model, ...]
        The sequence of models in parallel.
    solver : AbstractReducer, default=HierarchicalTreeReducer()
        The solver to use. Defaults to the hierarchical tree solver, which is 
        highly optimized for multi-port parallel merges.
    flatten: bool, default=True
        (experimental) Flattens nested Parallel instances into one large parallel group.

    Examples
    --------
    Create a parallel RLC tank circuit by connecting a resistor, 
    capacitor, and inductor in parallel:

    >>> import pmrf as prf
    >>> from pmrf.models import Resistor, Capacitor, Inductor, Parallel
    >>> 
    >>> res = Resistor(R=50.0)
    >>> cap = Capacitor(C=1e-12)
    >>> ind = Inductor(L=1e-9)
    >>> 
    >>> # The resulting model is a 2-port parallel RLC tank
    >>> rlc_tank = Parallel((res, cap, ind))
    """
    #: Input models to be placed in parallel
    parallel: InitVar[tuple[Model, ...]]
    
    #: Flatten the connections if they contain any sub-parallel groups
    flatten: InitVar[bool] = field(default=True, static=True, kw_only=True)
    
    #: The deduplicated models stored in the state
    models: tuple[Model, ...] = field(default=None, kw_only=True)
    
    #: The solver, defaulting to hierarchical block reduction
    solver: AbstractReducer = field(default=BlockSchurScatteringReducer())
    
    #: The collated indices defining the internal parallel nodes
    indexed_connections: list[list[tuple[int, int]]] = field(default=None, kw_only=True, static=True)
    
    #: The explicitly ordered absolute indices of the external ports
    port_index: list[int] = field(default=None, kw_only=True, static=True)

    def __post_init__(self, parallel: tuple[Model, ...], flatten: bool):
        if not parallel:
            raise ValueError("Parallel requires at least one model.")
            
        n_ports = parallel[0].nports
        for model in parallel:
            if model.nports != n_ports:
                raise ValueError(
                    f"All models must have exactly {n_ports} ports for a Parallel connection. "
                    f"Found a model with {model.nports} ports."
                )

        if flatten:
            merged = []
            for model in parallel:
                if isinstance(model, Parallel) and getattr(model, 'name', None) is None and getattr(model, 'metadata', None) is None:
                    merged.extend(model.models)
                else:
                    merged.append(model)
            parallel = tuple(merged)

        unique_models = []
        id_to_index = {}
        seen_ports = set()
        
        idx_conn = [[] for _ in range(n_ports)]

        for model in parallel:
            m_id = id(model)
            if m_id not in id_to_index:
                id_to_index[m_id] = len(unique_models)
                unique_models.append(model)
            
            m_idx = id_to_index[m_id]

            for p in range(n_ports):
                port_signature = (m_id, p)
                if port_signature in seen_ports:
                    raise ValueError(
                        f"Port {p} of model '{getattr(model, 'name', 'unnamed')}' is connected multiple times. "
                        "Identical model instances cannot be connected in parallel."
                    )
                seen_ports.add(port_signature)
                idx_conn[p].append((m_idx, p))

        self.models = tuple(unique_models)
        self.indexed_connections = idx_conn
        
        # Anchor the external ports to the absolute indices of the first unique model
        self.port_index = list(range(n_ports))

    @property
    def number_of_ports(self):
        return len(self.port_index)
    
    @property
    def topology(self) -> Topology:
        return Topology(
            components=list(self.models), 
            indexed_connections=self.indexed_connections, 
            port_indices=self.port_index, 
            ground_indices=[],
            marker_indices=[],
        )

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return reduce(self.topology, freq, self.solver, z0=z0).s

    def y(self, freq: Frequency) -> jnp.ndarray:
        return reduce(self.topology, freq, self.solver).y
    

class Connected(Model):
    """
    (experimental) Represents a direct port-to-port connection between one or two models.

    This container connects exactly two ports together. It handles both standard 
    connections (connecting port `a` on `model_a` to port `b` on `model_b`) and 
    inner connections (closing a loop by connecting two ports on the same model instance).

    The resultant external ports maintain the expected order: the unconnected ports 
    of `model_a` followed by the unconnected ports of `model_b`.

    Parameters
    ----------
    model_a : Model
        The first model.
    port_a : int
        The integer index of the port on `model_a` to connect.
    model_b : Model
        The second model. If `model_a` and `model_b` are the exact same instance, 
        an inner-connection is performed.
    port_b : int
        The integer index of the port on `model_b` to connect.
    solver : AbstractReducer, default=SubnetworkGrowthReducer()
        The solver to use. Defaults to :class:`pmrf.simulate.SubnetworkGrowthReducer`,
        which is optimized for pairwise connections.

    Examples
    --------
    Connect port 1 of a 2-port attenuator to port 0 of a 2-port amplifier:

    >>> import pmrf as prf
    >>> from pmrf.models import Attenuator, Amplifier, Connected
    >>> 
    >>> atten = Attenuator(loss=3.0)
    >>> amp = Amplifier(gain=10.0)
    >>> 
    >>> # The resulting model will have 2 external ports (atten port 0, amp port 1)
    >>> chain = Connected(atten, 1, amp, 0)
    
    Close a feedback loop by connecting port 2 to port 3 on a 4-port coupler:
    
    >>> coupler = DirectionalCoupler()
    >>> feedback = Connected(coupler, 2, coupler, 3)
    """
    #: The first model.
    model_a: InitVar[Model]
    #: Port index on the first model.
    port_a: int = field(static=True)
    
    #: The second model.
    model_b: InitVar[Model]
    #: Port index on the second model.
    port_b: int = field(static=True)

    #: The circuit solver. Defaults to iterative port elimination.
    solver: AbstractReducer = field(default=SequentialSchurScatteringReducer())

    #: The sequence of models forming this connection.
    models: tuple[Model, ...] = field(default=None, kw_only=True)
    
    #: The collated indices defining the single connection node.
    indexed_connections: list[list[tuple[int, int]]] = field(default=None, kw_only=True, static=True)
    
    #: The explicitly ordered indices of the remaining external ports.
    port_index: list[int] = field(default=None, kw_only=True, static=True)
    
    #: Indicates if this is an inner connection.
    is_inner: bool = field(default=False, kw_only=True, static=True)

    def __post_init__(self, model_a: Model, model_b: Model):
        # Input Validation
        if self.port_a < 0 or self.port_a >= model_a.nports:
            raise ValueError(
                f"port_a index {self.port_a} out of bounds for model_a (nports={model_a.nports})."
            )
        if self.port_b < 0 or self.port_b >= model_b.nports:
            raise ValueError(
                f"port_b index {self.port_b} out of bounds for model_b (nports={model_b.nports})."
            )

        # Detect Inner Connection vs Standard Connection
        if model_a is model_b:
            if self.port_a == self.port_b:
                raise ValueError("Cannot connect a port to itself.")
            
            # Store ONE instance to preserve weight-tying and architecture rules
            self.models = (model_a,)
            self.is_inner = True
            self.indexed_connections = [[(0, self.port_a), (0, self.port_b)]]
            
            # Retain all remaining ports in their original order
            self.port_index = [
                p for p in range(model_a.nports) 
                if p not in (self.port_a, self.port_b)
            ]
            
        else:
            # Store BOTH instances
            self.models = (model_a, model_b)
            self.is_inner = False
            self.indexed_connections = [[(0, self.port_a), (1, self.port_b)]]
            
            # Retain ports of A, followed by ports of B
            ports_a = [p for p in range(model_a.nports) if p != self.port_a]
            offset = model_a.nports
            ports_b = [offset + p for p in range(model_b.nports) if p != self.port_b]
            
            self.port_index = ports_a + ports_b

    @property
    def number_of_ports(self):
        return len(self.port_index)

    @property
    def topology(self) -> Topology:
        return Topology(
            components=list(self.models), 
            indexed_connections=self.indexed_connections, 
            port_indices=self.port_index, 
            ground_indices=[],
            marker_indices=[],
        )

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return reduce(self.topology, freq, self.solver, z0=z0).s

    def y(self, freq: Frequency) -> jnp.ndarray:
        return reduce(self.topology, freq, self.solver).y
    

class Cascade(Model):
    """
    Represents a cascade, or series connection, of two or more `Model` objects.

    This container connects multiple models end-to-end. The output port of
    one model is connected to the input port of the next.

    All models must have 2N-many ports. Ports N to 2*N-1 of the first model
    are connected to ports 0 to N-1 of the second, and so on.

    Any nested `Cascade` instances are automatically flattened to maintain
    a simple, linear chain of models.

    Parameters
    ----------
    cascade : tuple[Model]
        The sequence of models in the cascade.
    solver : AbstractCascader, default=RedhefferCascader()
        The solver to use. Available solvers can be found in :mod:`pmrf.simulate`.
    flatten: bool, default=True
        (experimental) Flattens the cascade into one large cascade if they contain sub-cascades.

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
    #: (experimental) Flatten the connections if they contain any sub-circuits
    flatten: InitVar[bool] = field(default=True, static=True, kw_only=True)
    
    #: The models.
    cascade: tuple[Model]
    
    #: The solver.
    solver: AbstractCascader = field(default=RedhefferScatteringCascader())
    
    def __post_init__(self, flatten: bool):
        for model in self.cascade:
            if model.nports % 2 != 0:
                raise ValueError('All networks must be 2N-ports for Cascade')
            
        if flatten:
            merged = []
            for model in self.cascade:
                # Only extend if the user has not given it a name or metadata
                if isinstance(model, Cascade) and model.name is None and model.metadata is None:
                    merged.extend(model.cascade)
                else:
                    merged.append(model)
            self.cascade = tuple(merged)
        

    @property
    def number_of_ports(self):
        return self.cascade[0].number_of_ports
    
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0):
        return cascade(self.cascade, frequency, solver=self.solver, z0=z0).s
    
    
class Terminated(Model):
    """
    Represents one network terminated in another.

    Parameters
    ----------

    terminated_from : Model
        The model being terminated.
    terminated_into : Model
        The model that `terminated_from` is terminated into.
    solver : AbstractCascader
        The solver to use. Available solvers can be found in :mod:`pmrf.simulate`.        
    """
    #: The "from" model.
    terminated_from: Model

    #: The "into" model.
    terminated_into: Model
    
    #: The solver.
    solver: AbstractTerminator = field(default=AnalyticScatteringTerminator())

    @property
    def number_of_ports(self):
        return self.terminated_from.number_of_ports - self.terminated_into.number_of_ports
    
    def __post_init__(self):
        if self.terminated_from.nports != 2*self.terminated_into.nports:
            raise ValueError("Terminated only supports terminating 2N port networks in a 1N port")
        
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return terminate(self.terminated_from, self.terminated_into, frequency, solver=self.solver, z0=z0).s