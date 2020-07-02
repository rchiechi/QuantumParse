"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from ase.build import molecule
from ase import units


def test_o2(cp2k_factory):
    calc = cp2k_factory.calc(
        label='test_O2', uks=True, cutoff=150 * units.Rydberg,
        basis_set="SZV-MOLOPT-SR-GTH")
    o2 = molecule('O2', calculator=calc)
    o2.center(vacuum=2.0)
    energy = o2.get_potential_energy()
    energy_ref = -861.057011375
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10
    print('passed test "O2"')
