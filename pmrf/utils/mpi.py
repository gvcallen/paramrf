from pmrf.constants import MPI_AVAILABLE, COMM

def wait_for_all_ranks():
    if not MPI_AVAILABLE:
        return
    COMM.Barrier()

def sync_across_all_ranks(x, root=0):
    if not MPI_AVAILABLE:
        return
    return COMM.bcast(x, root=root)