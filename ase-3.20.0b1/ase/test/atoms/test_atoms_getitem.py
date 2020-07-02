import pytest
from ase.atoms import Atoms


def test_atoms_getitem():
    w = Atoms('H2O',
              positions=[[2.264, 0.639, 0.876],
                         [0.792, 0.955, 0.608],
                         [1.347, 0.487, 1.234]],
              cell=[3, 3, 3],
              pbc=True)

    with pytest.raises(IndexError):
        w[True, False]

    assert(w[0, 1] == w[True, True, False])
    assert(w[0, 1] == w[0:2])
