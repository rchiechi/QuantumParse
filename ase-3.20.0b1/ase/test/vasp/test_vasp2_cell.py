def test_vasp2_cell(require_vasp):
    """

    Check the unit cell is handled correctly

    """

    import pytest
    from ase.test.vasp import installed2 as installed
    from ase.calculators.vasp import Vasp2 as Vasp
    from ase.build import molecule
    assert installed()


    # Molecules come with no unit cell

    atoms = molecule('CH4')
    calc = Vasp()

    with pytest.raises(ValueError):
        atoms.calc = calc
        atoms.get_total_energy()
