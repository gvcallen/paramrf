
def wait_for_all_ranks():
    from pmrf.constants import MPI_AVAILABLE, COMM
    
    if not MPI_AVAILABLE:
        return
    COMM.Barrier()

def sync_across_all_ranks(x, root=0):
    from pmrf.constants import MPI_AVAILABLE, COMM
    
    if not MPI_AVAILABLE:
        return
    return COMM.bcast(x, root=root)