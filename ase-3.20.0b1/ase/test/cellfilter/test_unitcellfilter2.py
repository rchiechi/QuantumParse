import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.test import gradient_test
from ase.constraints import UnitCellFilter, ExpCellFilter


@pytest.fixture
def setup_atoms(asap3):
    rng = np.random.RandomState(1)
    a0 = bulk('Cu', cubic=True)

    # perturb the atoms
    s = a0.get_scaled_positions()
    s[:, 0] *= 0.995
    a0.set_scaled_positions(s)

    # perturb the cell
    a0.cell += rng.uniform(-1e-1, 1e-2, size=(3, 3))

    a0.calc = asap3.EMT()
    return a0


def test_unitcellfilter(setup_atoms):
    ucf = UnitCellFilter(setup_atoms)
    f, fn = gradient_test(ucf)
    assert abs(f - fn).max() < 3e-6


def test_expcellfilter(setup_atoms):
    ecf = ExpCellFilter(setup_atoms)
    # test all derivatives
    f, fn = gradient_test(ecf)
    assert abs(f - fn).max() < 3e-6
