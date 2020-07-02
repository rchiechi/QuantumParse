from ase.build import bulk
from ase.spectrum.band_structure import calculate_band_structure


def test_bands(siesta_factory):
    atoms = bulk('Si')
    path = atoms.cell.bandpath('GXWK', density=10)
    atoms.calc = siesta_factory.calc(kpts=[2, 2, 2])
    bs = calculate_band_structure(atoms, path)
    print(bs)
    bs.write('bs.json')
