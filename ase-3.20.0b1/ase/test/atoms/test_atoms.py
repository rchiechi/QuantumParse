import pytest
import numpy as np
from ase import Atoms


def test_atoms():
    from ase import Atoms
    print(Atoms())
    print(Atoms('H2O'))
    #...


def test_numbers_input():
    numbers= np.array([[0, 1], [2, 3]])
    with pytest.raises(Exception, match='"numbers" must be 1-dimensional.'):
        Atoms(positions=np.zeros((2, 3)), numbers=numbers, cell=np.eye(3))

    Atoms(positions=np.zeros((2, 3)), numbers=[0, 1], cell=np.eye(3))
