"""
MPI implementation for OLE. This module is a wrapper around mpi4py.
It works if mpi4py is installed, but it is not required.

The implementation is based upon the COBAYA MPI implementation.

Credits to Jesus Torrado.
"""
import os
from typing import Any, Optional

# Vars to keep track of MPI parameters
_mpi: Any = None if os.environ.get('OLE_NOMPI', False) else -1
_mpi_size = -1
_mpi_comm: Any = -1
_mpi_rank: Optional[int] = -1


def get_mpi():
    """
    Import and returns the MPI object, or None if not running with MPI.

    Can be used as a boolean test if MPI is present.
    """
    global _mpi
    if _mpi == -1:
        try:
            from mpi4py import MPI
            _mpi = MPI

        except ImportError:
            _mpi = None
        else:
            if get_mpi_size()>1:
                try:
                    import dill
                except ImportError:
                    pass
                else:
                    _mpi.pickle.__init__(dill.dumps, dill.loads)

    return _mpi


def get_mpi_size():
    """
    Returns the number of MPI processes that have been invoked,
    or 0 if not running with MPI.
    """
    global _mpi_size
    if _mpi_size == -1:
        _mpi_size = getattr(get_mpi_comm(), "Get_size", lambda: 0)()
    return _mpi_size


def get_mpi_comm():
    """
    Returns the MPI communicator, or `None` if not running with MPI.
    """
    global _mpi_comm
    if _mpi_comm == -1:
        _mpi_comm = getattr(get_mpi(), "COMM_WORLD", None)
    return _mpi_comm

def get_mpi_rank():
    """
    Returns the rank of the current MPI process:
        * None: not running with MPI
        * Z>=0: process rank, when running with MPI

    Can be used as a boolean that returns `False` for both the root process,
    if running with MPI, or always for a single process; thus, everything under
    `if not(get_mpi_rank()):` is run only *once*.
    """
    global _mpi_rank
    if _mpi_rank == -1:
        _mpi_rank = getattr(get_mpi_comm(), "Get_rank", lambda: None)()
    if _mpi_rank is None:
        _mpi_rank = 0
    return _mpi_rank

def is_main_process():
    """
    Returns true if primary process or MPI not available.
    """
    return not bool(get_mpi_rank())