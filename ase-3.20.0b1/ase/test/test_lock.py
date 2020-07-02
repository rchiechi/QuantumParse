import os
import pytest
from ase.utils import Lock


def test_cannot_acquire_lock_twice():
    """Test timeout on Lock.acquire()."""

    lock = Lock('lockfile', timeout=0.3)
    with lock:
        with pytest.raises(TimeoutError):
            with lock:
                ...


def test_lock_close_file_descriptor():
    """Test that lock file descriptor is properly closed."""
    # The choice of timeout=1.0 is arbitrary but we don't want to use
    # something that is too large since it could mean that the test
    # takes long to fail.
    lock = Lock('lockfile', timeout=1.0)
    with lock:
        pass

    # If fstat raises OSError this means that fd.close() was
    # not called.
    with pytest.raises(OSError):
        os.fstat(lock.fd.name)
