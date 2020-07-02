import pytest
from ase.build import bulk
from ase.calculators.vasp import Vasp
from ase.test.vasp import installed


def test_vasp_net_charge(require_vasp):
    """
    Run VASP tests to ensure that determining number of electrons from
    user-supplied net charge (via the deprecated net_charge parameter) works
    correctly. This is conditional on the existence of the VASP_COMMAND or
    VASP_SCRIPT environment variables.

    This is mainly a slightly reduced duplicate of the vasp_charge test, but with
    flipped signs and with checks that ensure FutureWarning is emitted.

    Should be removed along with the net_charge parameter itself at some point.
    """

    assert installed()

    system = bulk('Al', 'fcc', a=4.5, cubic=True)

    # Dummy calculation to let VASP determine default number of electrons
    calc = Vasp(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False)
    calc.calculate(system)
    default_nelect_from_vasp = calc.get_number_of_electrons()
    assert default_nelect_from_vasp == 12

    # Compare VASP's output nelect from before + net charge to default nelect
    # determined by us + net charge
    with pytest.warns(FutureWarning):
        net_charge = -2
        calc = Vasp(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False,
                    net_charge=net_charge)
        calc.initialize(system)
        calc.write_input(system)
        calc.read_incar()
    assert calc.float_params['nelect'] == default_nelect_from_vasp + net_charge

    # Test that conflicts between explicitly given nelect and net charge are
    # detected
    with pytest.raises(ValueError):
        with pytest.warns(FutureWarning):
            calc = Vasp(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False,
                        nelect=default_nelect_from_vasp+net_charge+1,
                        net_charge=net_charge)
            calc.calculate(system)

    # Test that conflicts between charge and net_charge are detected
    with pytest.raises(ValueError):
        with pytest.warns(FutureWarning):
            calc = Vasp(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False,
                        charge=-net_charge-1, net_charge=net_charge)
            calc.calculate(system)

    # Test that nothing is written if net charge is 0 and nelect not given
    with pytest.warns(FutureWarning):
        calc = Vasp(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False,
                    net_charge=0)
        calc.initialize(system)
        calc.write_input(system)
        calc.read_incar()
    assert calc.float_params['nelect'] is None
