"""
Composite models that physically connect ports of other models.
"""

import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.models.components.ideal import Port
from pmrf.rf_functions.connections import connect_s_arbitrary, terminate_s_in_s, cascade_a, cascade_s
from pmrf.field import field

class Circuit(Model, transparent=True):
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
        super().__init__()
        if self.name is None:
            self.name = 'circuit'

        self.models = []
        self.indexed_connections = []
        self.port_idxs = []
        id_to_index: dict[int, int] = {}

        for connection in connections:
            indexed_connection = []
            for model, value in connection:
                model_id = id(model)
                if model_id not in id_to_index:
                    id_to_index[model_id] = len(self.models)
                    self.models.append(model)
                
                model_idx = id_to_index[model_id]
                indexed_connection.append((model_idx, value))
                
                if value > model.nports - 1:
                    raise ValueError(f"Port index out of bounds for model {model.name or model}")
            
            self.indexed_connections.append(indexed_connection)
            
        for model in self.models:
            if type(model) == Port: 
                self.port_idxs.append(id_to_index[id(model)])

    def s(self, freq: Frequency) -> jnp.array:
        Smats = [model.s(freq) for model in self.models]
        z0s = [model.z0 for model in self.models]

        Scon, _z0con = connect_s_arbitrary(Smats, z0s, self.indexed_connections, self.port_idxs)
        return Scon

class Cascade(Model, transparent=True):
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
        model_reduced = []
        for model in self.models:
            if model.nports % 2 != 0:
                raise ValueError('All networks must be 2N-ports for Cascade')
            if isinstance(model, Cascade):
                model_reduced.extend(model.models)
            else:
                model_reduced.append(model)
                
        # Generate numerically sequenced defaults (model_1, model_2, etc.)
        self.models = model_reduced

    def a(self, freq: Frequency) -> jnp.ndarray:
        return cascade_a([model.a(freq) for model in self.models])

    def s(self, freq: Frequency) -> jnp.ndarray:
        Smats = jnp.array([model.s(freq) for model in self.models])
        z0s = jnp.array([model.z0 for model in self.models])
        Scas, z0cas = cascade_s(Smats, z0s)
        return Scas
    
    
class Terminated(Model, transparent=True):
    """
    Represents one network terminated in another.
    """
    from_model: Model
    into_model: Model
    
    def __post_init__(self):
        if self.from_model.nports != 2*self.into_model.nports:
            raise ValueError("Currently, Terminated only supports 2-port networks terminated in a 1-port")

    def s(self, freq: Frequency) -> jnp.ndarray:
        Smat_from = self.from_model.s(freq)
        z0_from = self.from_model.z0
        Smat_into = self.into_model.s(freq)
        z0_into = self.into_model.z0
        S_term, z0_term = terminate_s_in_s(Smat_from, z0_from, Smat_into, z0_into)
        return S_term