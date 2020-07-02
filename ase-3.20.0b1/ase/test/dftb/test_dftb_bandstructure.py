import subprocess
from unittest import SkipTest

import pytest

from ase.build import bulk


@pytest.mark.skip('test is rather broken')
def test_dftb_bandstructure(dftb_factory):
    # We need to get the DFTB+ version to know
    # whether to skip this test or not.
    # For this, we need to run DFTB+ and grep
    # the version from the output header.
    #cmd = os.environ['ASE_DFTB_COMMAND'].split()[0]
    #cmd = dftb_factory.ex

    if 0:
        cmd = 'xxxxx'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        lines = ''
        for line in proc.stdout:
            l = line.decode()
            if 'DFTB+' in l and ('version' in l.lower() or 'release' in l.lower()):
                version = l[l.index('DFTB+'):]
                break
            lines += l + '\n'
        else:
            raise RuntimeError('Could not parse DFTB+ version ' + lines)

        if '17.1' not in version:
            msg = 'Band structure properties not present in results.tag for ' + version
            raise SkipTest(msg)

    # The actual testing starts here
    calc = dftb_factory.calc(
        label='dftb',
        kpts=(3,3,3),
        Hamiltonian_SCC='Yes',
        Hamiltonian_SCCTolerance=1e-5,
        Hamiltonian_MaxAngularMomentum_Si='d'
    )

    atoms = bulk('Si')
    atoms.calc = calc
    atoms.get_potential_energy()

    efermi = calc.get_fermi_level()
    assert abs(efermi - -2.90086680996455) < 1.

    # DOS does not currently work because of
    # missing "get_k_point_weights" function
    #from ase.dft.dos import DOS
    #dos = DOS(calc, width=0.2)
    #d = dos.get_dos()
    #e = dos.get_energies()
    #print(d, e)

    calc = dftb_factory.calc(
        atoms=atoms,
        label='dftb',
        kpts={'path':'WGXWLG', 'npoints':50},
        Hamiltonian_SCC='Yes',
        Hamiltonian_MaxSCCIterations=1,
        Hamiltonian_ReadInitialCharges='Yes',
        Hamiltonian_MaxAngularMomentum_Si='d'
    )

    atoms.calc = calc
    calc.calculate(atoms)

    #calc.results['fermi_levels'] = [efermi]
    calc.band_structure()
    # Maybe write the band structure or assert something?
