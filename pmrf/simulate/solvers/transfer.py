"""pmrf/simulate/transfer.py"""

import jax
import jax.numpy as jnp

from pmrf.simulate.base import AbstractTransferCascader, TransferResult

class TransferCascader(AbstractTransferCascader):
    def run(
        self, 
        a_stacked: jnp.ndarray, # Shape: (N_networks, N_ports, N_ports)
    ) -> TransferResult:
        if a_stacked.shape[0] == 1:
            return TransferResult(a=a_stacked[0])

        def scan_fn(carry, x):
            return carry @ x, None

        a_cas, _ = jax.lax.scan(
            scan_fn, 
            init=a_stacked[0], 
            xs=a_stacked[1:]
        )
        
        return TransferResult(a=a_cas)