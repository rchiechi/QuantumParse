"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from ase.build import molecule


def test_h2_ls(cp2k_factory):
    inp = """&FORCE_EVAL
               &DFT
                 &QS
                   LS_SCF ON
                 &END QS
               &END DFT
             &END FORCE_EVAL"""
    calc = cp2k_factory.calc(label='test_H2_LS', inp=inp)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    energy = h2.get_potential_energy()
    energy_ref = -30.6989581747
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 5e-7
    print('passed test "H2_LS"')
