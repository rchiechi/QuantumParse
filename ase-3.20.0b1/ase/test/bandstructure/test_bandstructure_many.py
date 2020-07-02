import pytest
from ase.calculators.test import FreeElectrons
from ase.lattice import all_variants
from ase.spectrum.band_structure import calculate_band_structure
from ase import Atoms


@pytest.mark.parametrize("i, lat",
                         [pytest.param(i, lat, id=lat.variant)
                          for i, lat in enumerate(all_variants())
                          if lat.ndim == 3])
def test_lattice_bandstructure(i, lat, figure):
    xid = '{:02d}.{}'.format(i, lat.variant)
    path = lat.bandpath(density=10)
    path.write('path.{}.json'.format(xid))
    atoms = Atoms(cell=lat.tocell(), pbc=True)
    atoms.calc = FreeElectrons(nvalence=0, kpts=path.kpts)
    bs = calculate_band_structure(atoms, path)
    bs.write('bs.{}.json'.format(xid))

    ax = figure.gca()
    bs.plot(ax=ax, emin=0, emax=20, filename='fig.{}.png'.format(xid))
