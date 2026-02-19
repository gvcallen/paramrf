import numpy as np
from typing import Any, Sequence
import jax
import jax.numpy as jnp
import pkgutil
import importlib
from datetime import datetime
from typing import Union, get_origin, get_args, Union
import sys
import logging
import inspect
from types import GenericAlias, UnionType
from equinox import field

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except:
    RANK = 0
    MPI_AVAILABLE = False

def wait_for_all_ranks():
    if not MPI_AVAILABLE:
        return
    COMM.Barrier()

def sync_across_all_ranks(x, root=0):
    if not MPI_AVAILABLE:
        return
    return COMM.bcast(x, root=root)