"""
Composite models that physically connect ports of other models.
"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import InitVar
from functools import cached_property

from pmrf.models.base import Model
from pmrf.models.components.ideal import Port, Ground
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.types import ArrayLike
from pmrf.rf import y2s, s2y

# Unified interfaces imported from base
from pmrf.models.composite.interconnected.base import (
    AbstractCircuitSolver,
    AbstractScatteringCircuitSolver,
    AbstractAdmittanceCircuitSolver,
    AbstractMNACircuitSolver,
    PortRepresentation,
    NodalRepresentation,
    MNARepresentation,
    ScatteringResult,
    AdmittanceResult,
)
from pmrf.models.composite.interconnected.solvers.scattering import (
    GlobalScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
)

EVAL_Z0 = 50.0

class Circuit(Model):
    """
    Represents an arbitrary interconnection of multiple `Model` objects.

    This container connects multiple models together based on a specified list
    of nodes. Each node connects one or more ports of the constituent models 
    to form a composite network.
    """
    #: The connections.
    connections: InitVar[list[list[tuple[Model, int]]]] = None
    
    #: Flattens the connections.
    flatten: InitVar[bool] = field(default=True, static=True, kw_only=True)
    
    #: The models in the connections.
    circuit: list[Model] = field(default=None, kw_only=True)

    #: The collated indices of the connections.
    indexed_connections: list[list[tuple[int, int]]] = field(default=None, kw_only=True, static=True)
    
    #: The circuit solver. (Requires default solver injected or specified at instantiation)
    solver: AbstractCircuitSolver = field(default_factory=GlobalScatteringCircuitSolver, kw_only=True)
    
    @staticmethod
    def _flatten_connections(connections):
        """Flattens nested Circuit and Cascade instances into base models."""
        model_map = {}
        nodes = []
        for connection in connections:
            node_set = set()
            for model, port in connection:
                model_map[id(model)] = model
                node_set.add((id(model), port))
            nodes.append(node_set)
            
        while True:
            sub_c_id = None
            sub_c = None
            for node in nodes:
                for mid, _ in node:
                    cls_name = model_map[mid].__class__.__name__
                    if cls_name in ('Circuit', 'Cascade'):
                        sub_c_id = mid
                        sub_c = model_map[mid]
                        break
                if sub_c_id: break
                
            if not sub_c_id:
                break 
                
            if sub_c.__class__.__name__ == 'Circuit':
                sub_ports = [m for m in sub_c.circuit if isinstance(m, Port)]
                for p in sub_ports:
                    model_map[id(p)] = p
                    
                for node in nodes:
                    items_to_remove, items_to_add = [], []
                    for item in node:
                        mid, port_idx = item
                        if mid == sub_c_id:
                            items_to_remove.append(item)
                            items_to_add.append((id(sub_ports[port_idx]), 0))
                    for item in items_to_remove: node.remove(item)
                    for item in items_to_add: node.add(item)
                        
                for idx_conn in sub_c.indexed_connections:
                    internal_node = set()
                    for m_idx, p_idx in idx_conn:
                        m = sub_c.circuit[m_idx]
                        model_map[id(m)] = m
                        internal_node.add((id(m), p_idx))
                    nodes.append(internal_node)

            elif sub_c.__class__.__name__ == 'Cascade':
                sub_models = sub_c.cascade
                for m in sub_models:
                    model_map[id(m)] = m
                    
                n_half = sub_models[0].nports // 2
                
                for node in nodes:
                    items_to_remove, items_to_add = [], []
                    for item in node:
                        mid, port_idx = item
                        if mid == sub_c_id:
                            items_to_remove.append(item)
                            if port_idx < n_half:
                                items_to_add.append((id(sub_models[0]), port_idx))
                            else:
                                items_to_add.append((id(sub_models[-1]), port_idx))
                                
                    for item in items_to_remove: node.remove(item)
                    for item in items_to_add: node.add(item)
                    
                for k in range(len(sub_models) - 1):
                    m_left, m_right = sub_models[k], sub_models[k+1]
                    for i in range(n_half):
                        internal_node = {
                            (id(m_left), n_half + i), 
                            (id(m_right), i)
                        }
                        nodes.append(internal_node)

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
            
            if sub_c.__class__.__name__ == 'Circuit':
                sub_port_ids = {id(p) for p in sub_ports}
                for node in nodes:
                    to_discard = [item for item in node if item[0] in sub_port_ids]
                    for item in to_discard:
                        node.discard(item)
                        
            nodes = [n for n in nodes if n]

        return [[(model_map[mid], port) for mid, port in n] for n in nodes]

    def __post_init__(self, connections: list, flatten: bool):
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
                    raise ValueError(f"Port index {value} out of bounds for model '{getattr(model, 'name', 'unnamed')}' (nports={model.nports}).")
                
                port_signature = (id(model), value)
                if port_signature in seen_ports:
                    raise ValueError(f"Port {value} of model '{getattr(model, 'name', 'unnamed')}' is connected multiple times.")
                seen_ports.add(port_signature)

        if flatten:
            connections = self._flatten_connections(connections)

        models = []
        indexed_connections = []
        id_to_index = {}

        for node_idx, connection in enumerate(connections):
            indexed_conn = []
            for item in connection:
                model, value = item
                model_id = id(model)
                if model_id not in id_to_index:
                    id_to_index[model_id] = len(models)
                    models.append(model)
                
                model_idx = id_to_index[model_id]
                indexed_conn.append((model_idx, value))
            
            indexed_connections.append(indexed_conn)
            
        self.circuit = models
        self.indexed_connections = indexed_connections

    @property
    def number_of_ports(self):
        return sum(1 for model in self.circuit if isinstance(model, Port))

    # --- TOPOLOGY REPRESENTATIONS ---

    @cached_property
    def port_representation(self) -> PortRepresentation:
        """Generates the static topological map for scattering connection and reduction."""
        def get_global_port(components: list[Model], c_idx: int, p_idx: int) -> int:
            return sum(c.nports for c in components[:c_idx]) + p_idx

        num_ports = sum(c.nports for c in self.circuit)
        parent = list(range(num_ports))
        
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        if self.indexed_connections:
            for cnx in self.indexed_connections:
                if not cnx: continue
                first = get_global_port(self.circuit, cnx[0][0], cnx[0][1])
                for c_idx, p_idx in cnx[1:]:
                    union(first, get_global_port(self.circuit, c_idx, p_idx))
                    
        port_to_net = np.array([find(i) for i in range(num_ports)], dtype=int)
        _, port_to_net_map = np.unique(port_to_net, return_inverse=True)

        ext_idx, int_idx = [], []
        offset = 0
        for model in self.circuit:
            for p in range(model.nports):
                if isinstance(model, Port):
                    ext_idx.append(offset + p)
                else:
                    int_idx.append(offset + p)
            offset += model.nports

        return PortRepresentation(
            ext_idx=np.array(ext_idx, dtype=int),
            int_idx=np.array(int_idx, dtype=int),
            port_to_net_map=port_to_net_map
        )
        
    @cached_property
    def nodal_representation(self) -> NodalRepresentation:
        """Generates the static topological map for nodal admittance assembly and reduction."""
        port_rep = self.port_representation
        port_to_net_map = port_rep.port_to_net_map
        
        r_idx, c_idx = [], []
        offset = 0
        for model in self.circuit:
            n = model.nports
            nets = port_to_net_map[offset:offset+n]
            R, C = np.meshgrid(nets, nets, indexing='ij')
            r_idx.append(R.flatten())
            c_idx.append(C.flatten())
            offset += n
            
        unique_ext_nets = np.unique(port_to_net_map[port_rep.ext_idx])
        unique_int_nets = np.setdiff1d(np.unique(port_to_net_map), unique_ext_nets)

        # Drop ground nodes (assumed to be nets belonging to a Ground component)
        ground_nets = set()
        offset = 0
        for model in self.circuit:
            if isinstance(model, Ground):
                ground_nets.update(port_to_net_map[offset:offset+model.nports])
            offset += model.nports
            
        unique_ext_nets = np.array([n for n in unique_ext_nets if n not in ground_nets], dtype=int)
        unique_int_nets = np.array([n for n in unique_int_nets if n not in ground_nets], dtype=int)

        return NodalRepresentation(
            r_idx=np.concatenate(r_idx).astype(int),
            c_idx=np.concatenate(c_idx).astype(int),
            ext_idx=unique_ext_nets,
            int_idx=unique_int_nets
        )

    @cached_property
    def mna_representation(self) -> MNARepresentation:
        """Generates the static topological map for MNA assembly and reduction."""
        nodal = self.nodal_representation
        
        y_r_idx, y_c_idx = nodal.r_idx, nodal.c_idx
        b_r_idx, b_c_idx = [], []
        c_r_idx, c_c_idx = [], []
        d_r_idx, d_c_idx = [], []
        
        aux_count = 0
        offset_port = 0
        for model in self.circuit:
            # We assume model.mna() returns a stamp where D.shape[1] defines aux count.
            # Here we just peek at the shape dynamically, or default to 0 if not MNA capable.
            # To remain static, we assume properties or a static peek into the model:
            k = getattr(model, 'mna_aux_count', 0) 
            n = model.nports
            
            if k > 0:
                nets = self.port_representation.port_to_net_map[offset_port:offset_port+n]
                aux_nets = np.arange(aux_count, aux_count + k) + np.max(self.port_representation.port_to_net_map) + 1
                
                BR, BC = np.meshgrid(nets, aux_nets, indexing='ij')
                CR, CC = np.meshgrid(aux_nets, nets, indexing='ij')
                DR, DC = np.meshgrid(aux_nets, aux_nets, indexing='ij')
                
                b_r_idx.append(BR.flatten())
                b_c_idx.append(BC.flatten())
                c_r_idx.append(CR.flatten())
                c_c_idx.append(CC.flatten())
                d_r_idx.append(DR.flatten())
                d_c_idx.append(DC.flatten())
                
            aux_count += k
            offset_port += n
            
        aux_start_idx = np.max(self.port_representation.port_to_net_map) + 1 if aux_count > 0 else 0
        aux_idx = np.arange(aux_start_idx, aux_start_idx + aux_count)

        def _safe_cat(lst):
            return np.concatenate(lst).astype(int) if lst else np.array([], dtype=int)

        return MNARepresentation(
            y_r_idx=y_r_idx, y_c_idx=y_c_idx,
            b_r_idx=_safe_cat(b_r_idx), b_c_idx=_safe_cat(b_c_idx),
            c_r_idx=_safe_cat(c_r_idx), c_c_idx=_safe_cat(c_c_idx),
            d_r_idx=_safe_cat(d_r_idx), d_c_idx=_safe_cat(d_c_idx),
            ext_idx=nodal.ext_idx,
            int_idx=nodal.int_idx,
            aux_idx=aux_idx
        )

    # --- DATA EVALUATION ---

    def _evaluate_scattering(self, freq: Frequency, z0: ArrayLike = 50.0) -> tuple[jnp.ndarray, jnp.ndarray]:
        S_blocks = [c.s(freq, z0=z0) for c in self.circuit]
        Nf = S_blocks[0].shape[0]
        num_ports = sum(S.shape[1] for S in S_blocks)
        dtype = S_blocks[0].dtype
        
        batched_S = jnp.zeros((Nf, num_ports, num_ports), dtype=dtype)
        offset = 0
        for S_c in S_blocks:
            n = S_c.shape[1]
            batched_S = batched_S.at[:, offset:offset+n, offset:offset+n].set(S_c)
            offset += n
            
        z0_ports = jnp.broadcast_to(jnp.asarray(z0, dtype=dtype), (num_ports,))
        return batched_S, z0_ports

    def _evaluate_admittance(self, freq: Frequency) -> jnp.ndarray:
        Y_blocks = [c.y(freq) for c in self.circuit]
        flat_Y_list = []
        for Y in Y_blocks:
            Nf, n, _ = Y.shape
            flat_Y_list.append(Y.reshape(Nf, n * n))
        return jnp.concatenate(flat_Y_list, axis=1)

    def _evaluate_mna(self, freq: Frequency) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        flat_Y, flat_B, flat_C, flat_D = [], [], [], []
        
        for c in self.circuit:
            stamp = c.mna(freq)
            Y, B, C, D = stamp.Y, stamp.B, stamp.C, stamp.D
            Nf, n, _ = Y.shape
            k = D.shape[1]
            
            flat_Y.append(Y.reshape(Nf, n * n))
            if k > 0:
                flat_B.append(B.reshape(Nf, n * k))
                flat_C.append(C.reshape(Nf, k * n))
                flat_D.append(D.reshape(Nf, k * k))
                
        def _safe_concat(array_list):
            return jnp.concatenate(array_list, axis=1) if array_list else jnp.zeros((freq.npoints, 0), dtype=jnp.complex128)
            
        return _safe_concat(flat_Y), _safe_concat(flat_B), _safe_concat(flat_C), _safe_concat(flat_D)

    # --- SIMULATION & CONVERSION ---

    def _solve(self, freq: Frequency, z0: ArrayLike = EVAL_Z0):
        """Dispatches data prep and solving across the active vmapped solver interface."""
        if isinstance(self.solver, AbstractScatteringCircuitSolver):
            s_bdiag, z0_ports = self._evaluate_scattering(freq, z0)
            run_vmap = jax.vmap(self.solver.run, in_axes=(0, None, None))
            return run_vmap(s_bdiag, z0_ports, self.port_representation)
            
        elif isinstance(self.solver, AbstractAdmittanceCircuitSolver):
            y_flat = self._evaluate_admittance(freq)
            run_vmap = jax.vmap(self.solver.run, in_axes=(0, None))
            return run_vmap(y_flat, self.nodal_representation)
            
        elif isinstance(self.solver, AbstractMNACircuitSolver):
            y_flat, b_flat, c_flat, d_flat = self._evaluate_mna(freq)
            run_vmap = jax.vmap(self.solver.run, in_axes=(0, 0, 0, 0, None))
            return run_vmap(y_flat, b_flat, c_flat, d_flat, self.mna_representation)
            
        else:
            raise TypeError(f"Unrecognized solver type: {type(self.solver)}")

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        result = self._solve(freq, z0)
        if isinstance(result, ScatteringResult):
            return result.s
        elif isinstance(result, AdmittanceResult):
            return y2s(result.y, z0=z0)
        else:
            raise ValueError(f"Got unknown circuit solver result type: {result}")

    def y(self, freq: Frequency) -> jnp.ndarray:
        result = self._solve(freq)
        if isinstance(result, ScatteringResult):
            return s2y(result.s, z0=result.z0)
        elif isinstance(result, AdmittanceResult):
            return result.y
        else:
            raise ValueError(f"Got unknown circuit solver result type: {result}")

    @classmethod
    def from_chain(cls, models: tuple[Model, ...], **kwargs):
        if not models:
            raise ValueError("A chain requires at least one model.")

        n_ports = models[0].nports
        if n_ports % 2 != 0:
            raise ValueError("The first model in the chain must have an even number of ports (2N).")
        n_half = n_ports // 2
        
        is_terminated = False
        for k, model in enumerate(models):
            if k == len(models) - 1:
                if model.nports == n_half:
                    is_terminated = True
                elif model.nports != n_ports:
                    raise ValueError(f"Last model must have {n_ports} or {n_half} ports.")
            else:
                if model.nports != n_ports:
                    raise ValueError(f"Intermediate model at index {k} must have exactly {n_ports} ports.")

        connections = []
        in_ports = [Port() for _ in range(n_half)]
        for i in range(n_half):
            connections.append([(in_ports[i], 0), (models[0], i)])

        for k in range(len(models) - 1):
            m_left, m_right = models[k], models[k+1]
            for i in range(n_half):
                connections.append([(m_left, n_half + i), (m_right, i)])

        if not is_terminated:
            out_ports = [Port() for _ in range(n_half)]
            for i in range(n_half):
                connections.append([(out_ports[i], 0), (models[-1], n_half + i)])

        return cls(connections=connections, **kwargs)
    
    @classmethod
    def from_connected(
        cls, 
        model_a: Model, 
        port_a: int, 
        model_b: Model, 
        port_b: int, 
        solver: AbstractCircuitSolver = None,
        **kwargs
    ):
        """
        Creates a Circuit representing a direct port-to-port connection between one or two models.

        This handles both standard connections (connecting port `a` on `model_a` to port `b` 
        on `model_b`) and inner connections (closing a loop by connecting two ports on the 
        same model instance).

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
        solver : AbstractCircuitSolver, optional
            The solver to use. Defaults to SequentialScatteringCircuitSolver.
        **kwargs
            Additional keyword arguments passed to the Circuit constructor.
            
        Examples
        --------
        Connect port 1 of a 2-port attenuator to port 0 of a 2-port amplifier:

        >>> atten, amp = Attenuator(loss=3.0), Amplifier(gain=10.0)
        >>> chain = Circuit.from_connected(atten, 1, amp, 0)
        """
        if solver is None:
            solver = SequentialScatteringCircuitSolver()

        if port_a < 0 or port_a >= model_a.nports:
            raise ValueError(f"port_a index {port_a} out of bounds for model_a (nports={model_a.nports}).")
        if port_b < 0 or port_b >= model_b.nports:
            raise ValueError(f"port_b index {port_b} out of bounds for model_b (nports={model_b.nports}).")

        connections = []

        if model_a is model_b:
            if port_a == port_b:
                raise ValueError("Cannot connect a port to itself.")
            # Inner connection node
            connections.append([(model_a, port_a), (model_a, port_b)])
            
            # Retain remaining ports in their original order
            unconnected = [(model_a, p) for p in range(model_a.nports) if p not in (port_a, port_b)]
        else:
            # Standard connection node
            connections.append([(model_a, port_a), (model_b, port_b)])
            
            # Retain ports of A, followed by ports of B
            unconnected = [(model_a, p) for p in range(model_a.nports) if p != port_a]
            unconnected += [(model_b, p) for p in range(model_b.nports) if p != port_b]

        # Wire up the unconnected ports to external Port() objects to expose them
        for m, p in unconnected:
            connections.append([(Port(), 0), (m, p)])

        return cls(connections=connections, solver=solver, **kwargs)

    @classmethod
    def from_parallel(
        cls, 
        models: tuple[Model, ...], 
        solver: AbstractCircuitSolver = None,
        **kwargs
    ):
        """
        Creates a Circuit representing a parallel connection of two or more `Model` objects.

        Port `i` of every model is connected to port `i` of every other model, forming 
        a single composite network with the same number of ports as the individual models.
        All models must have the exact same number of ports.

        Parameters
        ----------
        models : tuple[Model, ...]
            The sequence of models in parallel.
        solver : AbstractCircuitSolver, optional
            The solver to use. Defaults to HierarchicalScatteringCircuitSolver.
        **kwargs
            Additional keyword arguments passed to the Circuit constructor.

        Examples
        --------
        Create a parallel RLC tank circuit:

        >>> res, cap, ind = Resistor(50.0), Capacitor(1e-12), Inductor(1e-9)
        >>> rlc_tank = Circuit.from_parallel((res, cap, ind))
        """
        if solver is None:
            # Assuming you imported HierarchicalScatteringCircuitSolver
            solver = HierarchicalScatteringCircuitSolver()

        if not models:
            raise ValueError("Parallel requires at least one model.")
            
        n_ports = models[0].nports
        for k, model in enumerate(models):
            if model.nports != n_ports:
                raise ValueError(
                    f"All models must have exactly {n_ports} ports for a Parallel connection. "
                    f"Model at index {k} has {model.nports} ports."
                )

        connections = []
        
        # Tie each corresponding port index across all models to a single shared external Port
        for p in range(n_ports):
            node = [(Port(), 0)]
            for model in models:
                node.append((model, p))
            connections.append(node)

        return cls(connections=connections, solver=solver, **kwargs)