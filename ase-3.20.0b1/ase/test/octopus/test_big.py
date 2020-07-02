import pytest
import numpy as np

from ase.collections import g2
from ase.build import bulk, graphene_nanoribbon
from ase.calculators.interfacechecker import check_interface


def calculate(factory, system, **kwargs):
    calc = factory.calc(**kwargs)
    system.calc = calc
    system.get_potential_energy()
    calc.get_eigenvalues()
    check_interface(calc)
    return calc


calc = pytest.mark.calculator

@calc('octopus', Spacing='0.25 * angstrom')
@pytest.mark.xfail
def test_h2o(factory):
    calc = calculate(factory,
                     g2['H2O'],
                     OutputFormat='xcrysden',
                     SCFCalculateDipole=True)
    dipole = calc.get_dipole_moment()
    E = calc.get_potential_energy()

    print('dipole', dipole)
    print('energy', E)

    # XXX What's with the dipole not being correct?
    # XXX Investigate

    assert pytest.approx(dipole, abs=0.02) == [0, 0, -0.37]
    dipole_err = np.abs(dipole - [0., 0., -0.37]).max()
    assert dipole_err < 0.02, dipole_err
    #energy_err = abs(-463.5944954 - E)
    #assert energy_err < 0.01, energy_err

@calc('octopus', Spacing='0.2 * angstrom')
def test_o2(factory):
    atoms = g2['O2']
    atoms.center(vacuum=2.5)
    calculate(factory,
              atoms,
              BoxShape='parallelepiped',
              SpinComponents='spin_polarized',
              ExtraStates=2)
    #magmom = calc.get_magnetic_moment()
    #magmoms = calc.get_magnetic_moments()
    #print('magmom', magmom)
    #print('magmoms', magmoms)

@calc('octopus')
def test_si(factory):
    calc = calculate(factory,
                     bulk('Si'), #, orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     KPointsUseSymmetries=True,
                     SmearingFunction='fermi_dirac',
                     ExtraStates=2,
                     Smearing='0.1 * eV',
                     ExperimentalFeatures=True,
                     Spacing='0.45 * Angstrom')
    eF = calc.get_fermi_level()
    print('eF', eF)


if 0:  # This calculation does not run will in Octopus
    # We will do the "toothless" spin-polarised Si instead.
    calc = calculate('Fe',
                     bulk('Fe', orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     KPointsUseSymmetries=True,
                     ExtraStates=4,
                     Spacing='0.15 * Angstrom',
                     SmearingFunction='fermi_dirac',
                     Smearing='0.1 * eV',
                     PseudoPotentialSet='sg15',
                     ExperimentalFeatures=True,
                     SpinComponents='spin_polarized')
    eF = calc.get_fermi_level()
    assert abs(eF - 5.33) < 1e-1
    # XXXX octopus does not get magnetic state?
    print('eF', eF)

if 0:
    # Experimental feature: mixed periodicity.  Let us not do this for now...
    graphene = graphene_nanoribbon(2, 2, sheet=True)
    graphene.positions = graphene.positions[:, [0, 2, 1]]
    graphene.pbc = [1, 1, 0] # from 1, 0, 1
    calc = calculate('graphene',
                     graphene,
                     KPointsGrid=[[2, 1, 2]],
                     KPointsUseSymmetries=True,
                     ExperimentalFeatures=True,
                     ExtraStates=4,
                     SmearingFunction='fermi_dirac',
                     Smearing='0.1 * eV')
