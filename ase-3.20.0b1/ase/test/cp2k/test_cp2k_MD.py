"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet


def test_md(cp2k_factory):
    calc = cp2k_factory.calc(label='test_H2_MD')
    positions = [(0, 0, 0), (0, 0, 0.7245595)]
    atoms = Atoms('HH', positions=positions, calculator=calc)
    atoms.center(vacuum=2.0)

    # Run MD
    MaxwellBoltzmannDistribution(atoms, 0.5 * 300 * units.kB, force_temp=True)
    energy_start = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    dyn = VelocityVerlet(atoms, 0.5 * units.fs)
    #def print_md():
    #    energy = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    #    print("MD total-energy: %.10feV" %  energy)
    #dyn.attach(print_md, interval=1)
    dyn.run(20)

    energy_end = atoms.get_potential_energy() + atoms.get_kinetic_energy()

    assert energy_start - energy_end < 1e-4
    print('passed test "H2_MD"')
