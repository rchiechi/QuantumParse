def test_vasp_cell(require_vasp):
    """

    Check the unit cell is handled correctly

    """

    import pytest
    from ase.calculators.vasp import Vasp
    from ase.build import molecule

    # Molecules come with no unit cell

    atoms = molecule('CH4')
    calc = Vasp()

    with pytest.raises(RuntimeError):
        atoms.write('POSCAR')

    with pytest.raises(ValueError):
        atoms.calc = calc
        atoms.get_total_energy()
