import pytest
from ase.neighborlist import NeighborList
from ase.build import bulk


def test_neighborlist_initialization():

    atoms = bulk('Al', 'fcc', a=4)

    nl = NeighborList([8] * len(atoms), skin=0, self_interaction=False)
    with pytest.raises(Exception, match="Must call update"):
        nl.get_neighbors(0)

    with pytest.raises(Exception, match="Must call update"):
        nl.get_connectivity_matrix()

    nl.update(atoms)
    indices, offsets = nl.get_neighbors(0)
