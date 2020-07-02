"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from ase.build import molecule


def test_restart(cp2k_factory):
    calc = cp2k_factory.calc()
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    h2.get_potential_energy()
    calc.write('test_restart')  # write a restart
    calc2 = cp2k_factory.calc(restart='test_restart')  # load a restart
    assert not calc2.calculation_required(h2, ['energy'])
    print('passed test "restart"')
