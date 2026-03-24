"""
Composite models that physically connect ports of other models.
"""

import warnings
import jax.numpy as jnp
from parax import field
from dataclasses import InitVar

from pmrf.core import Model, Frequency
from pmrf.models.components.ideal import Port
from pmrf.rf_functions import connect_s_arbitrary, terminate_s_in_s, cascade_a, cascade_s

class Circuit(Model, transparent=True):
    # Inputs (init=True, but we don't need to keep them around in the PyTree)
    connections: InitVar[list[list[tuple[Model, int]]]] = None
    
    # Computed properties (init=False, they are generated, not passed in)
    models: list[Model] = field(init=False)
    indexed_connections: list[list[tuple[int, int]]] = field(init=False, static=True)
    port_idxs: list[int] = field(init=False, static=True)

    def __post_init__(self, connections):
        # Because the class is frozen, we use object.__setattr__ to assign computed values
        models = []
        indexed_connections = []
        port_idxs = []
        id_to_index = {}

        for connection in connections:
            indexed_conn = []
            for model, value in connection:
                model_id = id(model)
                if model_id not in id_to_index:
                    id_to_index[model_id] = len(models)
                    models.append(model)
                
                model_idx = id_to_index[model_id]
                indexed_conn.append((model_idx, value))
                
                if value > model.nports - 1:
                    raise ValueError(f"Port index out of bounds for model {model.name}")
            
            indexed_connections.append(indexed_conn)
            
        for model in models:
            if isinstance(model, Port): 
                port_idxs.append(id_to_index[id(model)])

        # Assign the computed values
        self.models = models
        self.indexed_connections = indexed_connections
        self.port_idxs = port_idxs

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
    >>> from pmrf.core import Resistor, Capacitor, Inductor

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


class Shunt(Model, transparent=True):
    r"""
    Represents a 1-port network connected in parallel (shunt) across a 2-port line.

    This maps the reflection coefficient ($\Gamma$ or $S_{11}$) of a 1-port 
    model into a 2-port transmission matrix. 

    Attributes
    ----------
    model : Model
        The 1-port model to be connected in shunt.
    """
    model: Model
    
    def __post_init__(self):
        if self.model.nports != 1:
            raise ValueError(f"Shunt requires a 1-port model. Received a {self.model.nports}-port model.")

    def s(self, freq: Frequency) -> jnp.ndarray:
        # Get the 1-port S-parameters. Shape: (npoints, 1, 1)
        # Note: This assumes self.model.z0 == self.z0. If your library allows 
        # mixed reference impedances, you will need to renormalize s_1p first.
        s_1p = self.model.s(freq)
        
        # Extract the reflection coefficient array
        gamma = s_1p[:, 0, 0]
        
        # Calculate 2-port S-parameters directly from 1-port Gamma
        # This avoids divide-by-zero errors for ideal opens/shorts
        denom = gamma + 3.0
        s11 = (gamma - 1.0) / denom
        s21 = 2.0 * (1.0 + gamma) / denom
        
        # Construct the (npoints, 2, 2) S-parameter array
        S_shunt = jnp.array([
            [s11, s21],
            [s21, s11],
        ]).transpose(2, 0, 1)
        
        return S_shunt