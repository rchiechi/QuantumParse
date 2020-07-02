"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from ase.build import molecule


def test_h2_lda(cp2k_factory):
    calc = cp2k_factory.calc(label='test_H2_LDA')
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    energy = h2.get_potential_energy()
    energy_ref = -30.6989595886
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10
    print('passed test "H2_LDA"')
