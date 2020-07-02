import numpy as np
import pytest

from ase.cluster.icosahedron import Icosahedron
from ase.neighborlist import neighbor_list

sym = 'Au'
a0 = 4.05
ico_cubocta_sizes = [1, 13, 55, 147, 309, 563, 923, 1415]
ico_corner_coordination = 6
ico_corners = 12


def coordination_numbers(atoms):
    return np.bincount(neighbor_list('i', atoms, 1.1 * a0))


@pytest.mark.parametrize('shells', range(1, 6))
def test_icosa(shells):
    atoms = Icosahedron(sym, shells)
    assert len(atoms) == ico_cubocta_sizes[shells - 1]

    coordination = coordination_numbers(atoms)
    if shells == 1:
        return

    assert min(coordination) == ico_corner_coordination
    ncorners = sum(coordination == ico_corner_coordination)
    assert ncorners == ico_corners
