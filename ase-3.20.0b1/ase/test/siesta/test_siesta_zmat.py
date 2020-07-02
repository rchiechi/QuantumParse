import os
from ase.constraints import FixAtoms, FixedLine, FixedPlane
from ase import Atoms


def test_siesta_zmat(siesta_factory):
    atoms = Atoms('CO2', [(0.0, 0.0, 0.0), (-1.178, 0.0, 0.0),
                          (1.178, 0.0, 0.0)])

    c1 = FixAtoms(indices=[0])
    c2 = FixedLine(1, [0.0, 1.0, 0.0])
    c3 = FixedPlane(2, [1.0, 0.0, 0.0])

    atoms.set_constraint([c1,c2,c3])

    custom_dir = './dir1/'

    # Test simple fdf-argument case.
    siesta = siesta_factory.calc(
        label=custom_dir + 'test_label',
        symlink_pseudos=False,
        atomic_coord_format='zmatrix',
        fdf_arguments={
            'MD.TypeOfRun': 'CG',
            'MD.NumCGsteps': 1000
            })

    atoms.calc = siesta
    siesta.write_input(atoms, properties=['energy'])

    with open(os.path.join(custom_dir, 'test_label.fdf'), 'r') as fd:
        lines = fd.readlines()
    lsl = [line.split() for line in lines]
    assert ['cartesian'] in lsl
    assert ['%block', 'Zmatrix'] in lsl
    assert ['%endblock', 'Zmatrix'] in lsl
    assert ['MD.TypeOfRun', 'CG'] in lsl
    assert any([line.split()[4:9] == ['0', '0', '0', '1', 'C'] for line in lines])
    assert any([line.split()[4:9] == ['0', '1', '0', '2', 'O'] for line in lines])
    assert any([line.split()[4:9] == ['0', '1', '1', '3', 'O'] for line in lines])
