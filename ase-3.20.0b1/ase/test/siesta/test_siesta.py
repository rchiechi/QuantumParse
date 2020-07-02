import numpy as np

from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase import Atoms


def test_siesta(siesta_factory):
    # Setup test structures.
    h = Atoms('H', [(0.0, 0.0, 0.0)])
    ch4 = Atoms('CH4', np.array([
        [0.000000, 0.000000, 0.000000],
        [0.682793, 0.682793, 0.682793],
        [-0.682793, -0.682793, 0.682790],
        [-0.682793, 0.682793, -0.682793],
        [0.682793, -0.682793, -0.682793]]))

    siesta = siesta_factory.calc()

    # Test simple fdf-argument case.
    atoms = h.copy()
    siesta = siesta_factory.calc(
        label='test_label',
        fdf_arguments={'DM.Tolerance': 1e-3})
    atoms.calc = siesta
    siesta.write_input(atoms, properties=['energy'])
    atoms = h.copy()
    atoms.calc = siesta
    siesta.write_input(atoms, properties=['energy'])
    with open('test_label.fdf', 'r') as fd:
        lines = fd.readlines()
    assert any([line.split() == ['DM.Tolerance', '0.001'] for line in lines])

    # Test (slightly) more complex case of setting fdf-arguments.
    siesta = siesta_factory.calc(
        label='test_label',
        mesh_cutoff=3000,
        fdf_arguments={
            'DM.Tolerance': 1e-3,
            'ON.eta': (5, 'Ry')})
    atoms.calc = siesta
    siesta.write_input(atoms, properties=['energy'])
    atoms = h.copy()
    atoms.calc = siesta
    siesta.write_input(atoms, properties=['energy'])
    with open('test_label.fdf', 'r') as f:
        lines = f.readlines()

    assert 'MeshCutoff\t3000\teV\n' in lines
    assert 'DM.Tolerance\t0.001\n' in lines
    assert 'ON.eta\t5\tRy\n' in lines

    # Test setting fdf-arguments after initiation.
    siesta.set_fdf_arguments(
        {'DM.Tolerance': 1e-2,
         'ON.eta': (2, 'Ry')})
    siesta.write_input(atoms, properties=['energy'])
    with open('test_label.fdf', 'r') as f:
        lines = f.readlines()
    assert 'MeshCutoff\t3000\teV\n' in lines
    assert 'DM.Tolerance\t0.01\n' in lines
    assert 'ON.eta\t2\tRy\n' in lines

    # Test initiation using Species.
    atoms = ch4.copy()
    species, numbers = siesta.species(atoms)
    assert all(numbers == np.array([1, 2, 2, 2, 2]))
    siesta = siesta_factory.calc(species=[Species(symbol='C', tag=1)])
    species, numbers = siesta.species(atoms)
    assert all(numbers == np.array([1, 2, 2, 2, 2]))
    atoms.set_tags([0, 0, 0, 1, 0])
    species, numbers = siesta.species(atoms)
    assert all(numbers == np.array([1, 2, 2, 2, 2]))
    siesta = siesta_factory.calc(species=[Species(symbol='H', tag=1, basis_set='SZ')])
    species, numbers = siesta.species(atoms)
    assert all(numbers == np.array([1, 2, 2, 3, 2]))
    siesta = siesta_factory.calc(label='test_label', species=species)
    siesta.write_input(atoms, properties=['energy'])
    with open('test_label.fdf', 'r') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    assert ['1', '6', 'C.lda.1'] in lines
    assert ['2', '1', 'H.lda.2'] in lines
    assert ['3', '1', 'H.lda.3'] in lines
    assert ['C.lda.1', 'DZP'] in lines
    assert ['H.lda.2', 'DZP'] in lines
    assert ['H.lda.3', 'SZ'] in lines

    # Test if PAO block can be given as species.
    c_basis = """2 nodes 1.00
    0 1 S 0.20 P 1 0.20 6.00
    5.00
    1.00
    1 2 S 0.20 P 1 E 0.20 6.00
    6.00 5.00
    1.00 0.95"""
    basis_set = PAOBasisBlock(c_basis)
    species = Species(symbol='C', basis_set=basis_set)
    siesta = siesta_factory.calc(label='test_label', species=[species])
    siesta.write_input(atoms, properties=['energy'])
    with open('test_label.fdf', 'r') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    assert ['%block', 'PAO.Basis'] in lines
    assert ['%endblock', 'PAO.Basis'] in lines
