from __future__ import print_function, division
import atexit
import functools
import pickle
import sys
import time

import numpy as np

from ase.utils import devnull


def get_txt(txt, rank):
    if hasattr(txt, 'write'):
        # Note: User-supplied object might write to files from many ranks.
        return txt
    elif rank == 0:
        if txt is None:
            return devnull
        elif txt == '-':
            return sys.stdout
        else:
            return open(txt, 'w', 1)
    else:
        return devnull


def paropen(name, mode='r', buffering=-1):
    """MPI-safe version of open function.

    In read mode, the file is opened on all nodes.  In write and
    append mode, the file is opened on the master only, and /dev/null
    is opened on all other nodes.
    """
    if world.rank > 0 and mode[0] != 'r':
        name = '/dev/null'
    return open(name, mode, buffering)


def parprint(*args, **kwargs):
    """MPI-safe print - prints only from master. """
    if world.rank == 0:
        print(*args, **kwargs)


class DummyMPI:
    rank = 0
    size = 1

    def sum(self, a):
        if isinstance(a, np.ndarray) and a.ndim > 0:
            pass
        else:
            return a

    def barrier(self):
        pass

    def broadcast(self, a, rank):
        pass


class MPI4PY:
    def __init__(self):
        from mpi4py import MPI
        self.comm_world = MPI.COMM_WORLD
        self.comm = self.comm_world
        self.rank = self.comm.rank
        self.size = self.comm.size

    def sum(self, a):
        return self.comm.allreduce(a)

    def split(self, split_size=None):
        """Divide the communicator."""
        # color - subgroup id
        # key - new subgroup rank
        if not split_size:
            split_size = self.size
        color = int(self.rank // (self.size / split_size))
        key = int(self.rank % (self.size / split_size))
        self.comm = self.comm.Split(color, key)
        self.rank = self.comm.rank
        self.size = self.comm.size

    def barrier(self):
        self.comm.barrier()

    def abort(self, code):
        self.comm.Abort(code)

    def broadcast(self, a, rank):
        a[:] = self.comm.bcast(a, root=rank)


# Check for special MPI-enabled Python interpreters:
if '_gpaw' in sys.builtin_module_names:
    # http://wiki.fysik.dtu.dk/gpaw
    import _gpaw
    world = _gpaw.Communicator()
elif '_asap' in sys.builtin_module_names:
    # Modern version of Asap
    # http://wiki.fysik.dtu.dk/asap
    # We cannot import asap3.mpi here, as that creates an import deadlock
    import _asap
    world = _asap.Communicator()
elif 'asapparallel3' in sys.modules:
    # Older version of Asap
    import asapparallel3
    world = asapparallel3.Communicator()
elif 'Scientific_mpi' in sys.modules:
    from Scientific.MPI import world
elif 'mpi4py' in sys.modules:
    world = MPI4PY()
else:
    # This is a standard Python interpreter:
    world = DummyMPI()

rank = world.rank
size = world.size
barrier = world.barrier


def broadcast(obj, root=0, comm=world):
    """Broadcast a Python object across an MPI communicator and return it."""
    if comm.rank == root:
        string = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        n = np.array([len(string)], int)
    else:
        string = None
        n = np.empty(1, int)
    comm.broadcast(n, root)
    if comm.rank == root:
        string = np.fromstring(string, np.int8)
    else:
        string = np.zeros(n, np.int8)
    comm.broadcast(string, root)
    if comm.rank == root:
        return obj
    else:
        return pickle.loads(string.tostring())


def parallel_function(func):
    """Decorator for broadcasting from master to slaves using MPI."""
    if world.size == 1:
        return func

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # Hook to disable.  Use self.serial = True
        if args and getattr(args[0], 'serial', False):
            return func(*args, **kwargs)
        ex = None
        result = None
        if world.rank == 0:
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                pass
        ex, result = broadcast((ex, result))
        if ex is not None:
            raise ex
        return result
    return new_func


def parallel_generator(generator):
    """Decorator for broadcasting yields from master to slaves using MPI."""
    if world.size == 1:
        return generator

    @functools.wraps(generator)
    def new_generator(*args, **kwargs):
        # Hook to disable.  Use self.serial = True
        if args and getattr(args[0], 'serial', False):
            for result in generator(*args, **kwargs):
                yield result
            return
        if world.rank == 0:
            try:
                for result in generator(*args, **kwargs):
                    broadcast((None, result))
                    yield result
            except Exception as ex:
                broadcast((ex, None))
                raise ex
            broadcast((None, None))
        else:
            ex, result = broadcast((None, None))
            if ex is not None:
                raise ex
            while result is not None:
                yield result
                ex, result = broadcast((None, None))
                if ex is not None:
                    raise ex
    return new_generator


def register_parallel_cleanup_function():
    """Call MPI_Abort if python crashes.

    This will terminate the processes on the other nodes."""

    if size == 1:
        return

    def cleanup(sys=sys, time=time, world=world):
        error = getattr(sys, 'last_type', None)
        if error:
            sys.stdout.flush()
            sys.stderr.write(('ASE CLEANUP (node %d): %s occurred.  ' +
                              'Calling MPI_Abort!\n') % (world.rank, error))
            sys.stderr.flush()
            # Give other nodes a moment to crash by themselves (perhaps
            # producing helpful error messages):
            time.sleep(3)
            world.abort(42)

    atexit.register(cleanup)


def distribute_cpus(size, comm):
    """Distribute cpus to tasks and calculators.

    Input:
    size: number of nodes per calculator
    comm: total communicator object

    Output:
    communicator for this rank, number of calculators, index for this rank
    """

    assert size <= comm.size
    assert comm.size % size == 0

    tasks_rank = comm.rank // size

    r0 = tasks_rank * size
    ranks = np.arange(r0, r0 + size)
    mycomm = comm.new_communicator(ranks)

    return mycomm, comm.size // size, tasks_rank
