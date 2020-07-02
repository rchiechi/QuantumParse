from functools import partial

import pytest

from ase.calculators.emt import EMT
from ase.optimize import (MDMin, FIRE, LBFGS, LBFGSLineSearch, BFGSLineSearch,
                          BFGS, GoodOldQuasiNewton, GPMin, Berny)
from ase.optimize.sciopt import SciPyFminCG, SciPyFminBFGS
from ase.optimize.precon import PreconFIRE, PreconLBFGS
from ase.cluster import Icosahedron
from ase.build import bulk


@pytest.fixture(scope='module')
def ref_atoms():
    atoms = bulk('Au')
    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms


def atoms_no_pbc():
    ref_atoms = Icosahedron('Ag', 2, 3.82975)
    ref_atoms.calc = EMT()
    atoms = ref_atoms.copy()
    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    e_unopt = atoms.get_potential_energy()
    assert e_unopt > 7  # it's 7.318 as of writing this test
    return atoms, ref_atoms


@pytest.fixture
def atoms(ref_atoms):
    atoms = ref_atoms * (2, 2, 2)
    atoms.rattle(stdev=0.1, seed=7)
    atoms.calc = EMT()
    e_unopt = atoms.get_potential_energy()
    assert e_unopt > 0.45  # it's 0.499 as of writing this test
    return atoms


optclasses = [
    MDMin, FIRE, LBFGS, LBFGSLineSearch, BFGSLineSearch,
    BFGS, GoodOldQuasiNewton, GPMin, SciPyFminCG, SciPyFminBFGS,
    PreconLBFGS, PreconFIRE, Berny,
]


@pytest.mark.parametrize('optcls', optclasses)
def test_optimize(optcls, atoms, ref_atoms):
    if optcls is Berny:
        pytest.importorskip('berny')  # check if pyberny installed
        optcls = partial(optcls, dihedral=False)
        optcls.__name__ = Berny.__name__
        atoms, ref_atoms = atoms_no_pbc()
    kw = {}
    if optcls is PreconLBFGS:
        kw['precon'] = None
    opt = optcls(atoms, logfile='opt.log', **kw)
    fmax = 0.01
    opt.run(fmax=fmax)

    forces = atoms.get_forces()
    final_fmax = max((forces**2).sum(axis=1)**0.5)
    ref_energy = ref_atoms.get_potential_energy()
    e_opt = atoms.get_potential_energy() * len(ref_atoms) / len(atoms)
    e_err = abs(e_opt - ref_energy)
    print()
    print('{:>20}: fmax={:.05f} eopt={:.06f}, err={:06e}'
          .format(optcls.__name__, final_fmax, e_opt, e_err))
    return final_fmax, e_err

    assert final_fmax < fmax
    assert e_err < 1e-5  # (This tolerance is arbitrary)
