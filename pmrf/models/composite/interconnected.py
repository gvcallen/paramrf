"""
Composite models that physically connect ports of other models.
"""
import jax.numpy as jnp
from dataclasses import InitVar

from pmrf.models import Model, Ground
from pmrf.frequency import Frequency
from pmrf.utils import field, ArrayLike
from pmrf.models.components.ideal import Port
from pmrf.rf import connect_s_arbitrary, connect_y_arbitrary, cascade_a, cascade_s, terminate_s_in_s, terminate_a_in_s, renormalize_s

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
    
    #: The models in the circuit.
    circuit: list[Model] = field(default=None, kw_only=True)

    #: The indices of the connections.
    indexed_connections: list[list[tuple[int, int]]] = field(default=None, kw_only=True, static=True)

    #: The indices of the ports.
    port_idxs: list[int] = field(default=None, kw_only=True, static=True)
    
    #: (experimental) The domain to perform the calculation in.
    #: Options are ('s', 'y'), where 'y' is experimental.
    domain: str = field(default='s', kw_only=True, static=True)
    
    #: The algorithm to use for the call to `connect_<domain>_arbitrary`.
    #: If None, uses the default algorithm for the domain.
    #: Algorithms are available in :mod:`pmrf.rf`.
    method: str | None = field(default=None, kw_only=True, static=True)

    def __post_init__(self, connections):
        if not isinstance(connections, list):
            raise TypeError("`connections` must be a list of lists (representing nodes).")

        models = []
        indexed_connections = []
        port_idxs = []
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
            
        for model in models:
            if isinstance(model, Port): 
                port_idxs.append(id_to_index[id(model)])

        # Assign the computed values
        self.circuit = models
        self.indexed_connections = indexed_connections
        self.port_idxs = port_idxs
        
    @property
    def primary_domain(self):
        return self.domain
        
    def primary_matrix(self, frequency: Frequency, **kwargs) -> jnp.ndarray:
        if self.domain == 's':
            z0 = kwargs.pop('z0')
            return self.s_impl(frequency, z0=z0)
        elif self.domain == 'y':
            return self.y_impl(frequency)
        else:
            raise Exception(f"No circuit connection algorithms available for the specified '{self.domain}' domain")
    
    def s_impl(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        Smats = [model.s(freq, z0=EVAL_Z0) for model in self.circuit]
        z0s = [EVAL_Z0 for _ in self.circuit]
        
        kwargs = {'method': self.method} if self.method is not None else {}
        Scon, _ = connect_s_arbitrary(
            Smats,
            z0s,
            self.indexed_connections,
            self.port_idxs,
            **kwargs,
        )
        
        return renormalize_s(Scon, z_old=EVAL_Z0, z_new=z0, s_def_old='power', s_def_new='power')    
    
    def y_impl(self, freq: Frequency) -> jnp.ndarray:
        """
        Evaluate the Y-parameters of the composite circuit.
        """
        Ymats = []
        for model in self.circuit:
            if isinstance(model, Ground):
                Ymats.append(jnp.zeros((freq.npoints, 1, 1), dtype=jnp.complex128))
            elif isinstance(model, Port):
                Ymats.append(jnp.zeros((freq.npoints, model.nports, model.nports), dtype=jnp.complex128))
            else:
                Ymats.append(model.y(freq))

        # We need to explicitly pass grounded nodes
        ground_idxs = [idx for idx, model in enumerate(self.circuit) if isinstance(model, Ground)]
        grounded_nodes = set()
        for node_idx, connection in enumerate(self.indexed_connections):
            for model_idx, port_idx in connection:
                if model_idx in ground_idxs:
                    grounded_nodes.add(node_idx)
                    
        kwargs = {'method': self.method} if self.method is not None else {}
        Ycon = connect_y_arbitrary(
            Ymats=Ymats, 
            connections=self.indexed_connections, 
            port_indices=self.port_idxs,
            grounded_nodes=list(grounded_nodes),
            **kwargs,
        )
        return Ycon

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
    
    #: The domain to perform the calculation in. Only 's' is supported.
    domain: str = field(default='s', kw_only=True, static=True)
    
    #: The algorithm to use for the call to `cascade_<domain>`.
    #: If None, uses the default algorithm for the domain.
    #: Algorithms are available in :mod:`pmrf.rf`.
    method: str | None = field(default=None, kw_only=True, static=True)    
    
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
    
    @property
    def primary_domain(self):
        return self.domain
    
    def primary_matrix(self, frequency: Frequency, **kwargs):
        if self.domain == 's':
            z0 = kwargs.pop('z0')
            return self.s_impl(frequency, z0=z0)
        elif self.domain == 'a':
            return self.a_impl(frequency)
        else:
            raise ValueError(f"No circuit connection algorithms available for the specified '{self.domain}' domain")

    def a_impl(self, frequency: Frequency) -> jnp.ndarray:
        kwargs = {'method': self.method} if self.method is not None else {}
        return cascade_a([model.a(frequency) for model in self.merged_cascade], **kwargs)

    def s_impl(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        merged_models = self.merged_cascade
        
        Smats = jnp.array([model.s(frequency, z0=EVAL_Z0) for model in merged_models])
        z0s = jnp.array([EVAL_Z0 for _ in merged_models])
        
        kwargs = {'method': self.method} if self.method is not None else {}
        Scas, _ = cascade_s(Smats, z0s, **kwargs)
        
        return renormalize_s(Scas, z_old=EVAL_Z0, z_new=z0, s_def_old='power', s_def_new='power')
    
    
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
    
    #: The domain to use for the "from" model. Only 's' is supported.
    domain_from: str = field(default='s', kw_only=True, static=True)
    
    #: The domain to use for the "to" model. Only 's' is supported.
    domain_to: str = field(default='s', kw_only=True, static=True)
    
    #: The algorithm to use for the call to `terminate_<domain_from>_in_<domain_to>`.
    #: If None, uses the default algorithm for the domain combination.
    #: Algorithms are available in :mod:`pmrf.rf`.
    method: str | None = field(default=None, kw_only=True, static=True)    
    
    def __post_init__(self):
        if self.terminated_from.nports != 2*self.terminated_into.nports:
            raise ValueError("Terminated only supports terminating 2N port networks in a 1N port")
        
    @property
    def primary_domain(self):
        if self.domain_from == 'a' and self.domain_to == 's':
            return 's'
        elif self.domain_from == 's' and self.domain_to == 's':
            return 's'
        else:
            raise ValueError(f"Invalid combination of domains in Terminated: from: '{self.domain_from}', to: '{self.domain_to}'")
    
    def primary_matrix(self, frequency: Frequency, **kwargs):
        if self.domain_from == 'a' and self.domain_to == 's':
            z0 = kwargs.pop('z0')
            return self.a_in_s_impl(frequency, z0=z0)
        elif self.domain_from == 's' and self.domain_to == 's':
            z0 = kwargs.pop('z0')
            return self.s_in_s_impl(frequency, z0=z0)
        else:
            raise ValueError(f"Invalid combination of domains in Terminated: from: '{self.domain_from}', to: '{self.domain_to}'")
        
    def a_in_s_impl(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        Amat_from = self.terminated_from.a(frequency)
        Smat_into = self.terminated_into.s(frequency, z0=z0)
        
        kwargs = {'method': self.method} if self.method is not None else {}
        S_term = terminate_a_in_s(Amat_from, Smat_into, z0=z0, s_def='power', **kwargs)
        
        return S_term        

    def s_in_s_impl(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        Smat_from = self.terminated_from.s(frequency, z0=EVAL_Z0)
        Smat_into = self.terminated_into.s(frequency, z0=EVAL_Z0)
        
        kwargs = {'method': self.method} if self.method is not None else {}
        S_term, _ = terminate_s_in_s(Smat_from=Smat_from, z0_from=EVAL_Z0, Smat_into=Smat_into, z0_into=EVAL_Z0, **kwargs)
        
        return renormalize_s(S_term, EVAL_Z0, z0, 'power', 'power')