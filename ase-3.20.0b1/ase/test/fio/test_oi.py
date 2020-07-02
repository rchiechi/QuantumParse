import warnings

import pytest
import numpy as np

from ase import Atoms
from ase.io import write, read, iread
from ase.io.formats import all_formats, ioformats
from ase.calculators.singlepoint import SinglePointCalculator

try:
    import matplotlib
except ImportError:
    matplotlib = 0

try:
    import netCDF4
except ImportError:
    netCDF4 = 0


@pytest.fixture
def atoms():
    a = 5.0
    d = 1.9
    c = a / 2
    atoms = Atoms('AuH',
                  positions=[(0, c, c), (d, c, c)],
                  cell=(2 * d, a, a),
                  pbc=(1, 0, 0))
    extra = np.array([2.3, 4.2])
    atoms.set_array('extra', extra)
    atoms *= (2, 1, 1)

    # attach some results to the Atoms.
    # These are serialised by the extxyz writer.

    spc = SinglePointCalculator(atoms,
                                energy=-1.0,
                                stress=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                forces=-1.0 * atoms.positions)
    atoms.calc = spc
    return atoms


def check(a, ref_atoms, format):
    assert abs(a.positions - ref_atoms.positions).max() < 1e-6, \
        (a.positions - ref_atoms.positions)
    if format in ['traj', 'cube', 'cfg', 'struct', 'gen', 'extxyz',
                  'db', 'json', 'trj']:
        assert abs(a.cell - ref_atoms.cell).max() < 1e-6
    if format in ['cfg', 'extxyz']:
        assert abs(a.get_array('extra') -
                   ref_atoms.get_array('extra')).max() < 1e-6
    if format in ['extxyz', 'traj', 'trj', 'db', 'json']:
        assert (a.pbc == ref_atoms.pbc).all()
        assert a.get_potential_energy() == ref_atoms.get_potential_energy()
        assert (a.get_stress() == ref_atoms.get_stress()).all()
        assert abs(a.get_forces() - ref_atoms.get_forces()).max() < 1e-12


@pytest.fixture
def catch_warnings():
    return warnings.catch_warnings()


def all_tested_formats():
    skip = []

    # Someone should do something ...
    skip += ['abinit', 'castep-cell', 'dftb', 'eon', 'gaussian', 'lammps-data']

    # Standalone test used as not compatible with 1D periodicity
    skip += ['v-sim', 'mustem', 'prismatic']

    # We have a standalone dmol test
    skip += ['dmol-arc', 'dmol-car', 'dmol-incoor']

    # Complex dependencies; see animate.py test
    skip += ['gif', 'mp4']

    # Let's not worry about these.
    skip += ['postgresql', 'trj', 'vti', 'vtu', 'mysql']

    if not matplotlib:
        skip += ['eps', 'png']

    if not netCDF4:
        skip += ['netcdftrajectory']

    return sorted(set(all_formats) - set(skip))


@pytest.mark.parametrize('format', all_tested_formats())
def test_ioformat(format, atoms, catch_warnings):
    if format in ['proteindatabank', 'netcdftrajectory']:
        warnings.simplefilter('ignore', UserWarning)

    if format == 'dlp4':
        atoms.pbc = (1, 1, 0)

    images = [atoms, atoms]

    io = ioformats[format]
    print('{0:20}{1}{2}{3}{4}'.format(format,
                                      ' R'[bool(io.read)],
                                      ' W'[bool(io.write)],
                                      '+1'[io.single],
                                      'SF'[io.acceptsfd]))
    fname1 = 'io-test.1.{}'.format(format)
    fname2 = 'io-test.2.{}'.format(format)
    if io.write:
        write(fname1, atoms, format=format)
        if not io.single:
            write(fname2, images, format=format)

        if io.read:
            for a in [read(fname1, format=format), read(fname1)]:
                check(a, atoms, format)

            if not io.single:
                if format in ['json', 'db']:
                    aa = read(fname2, index='id=1') + read(fname2, index='id=2')
                else:
                    aa = [read(fname2), read(fname2, 0)]
                aa += read(fname2, ':')
                for a in iread(fname2, format=format):
                    aa.append(a)
                assert len(aa) == 6, aa
                for a in aa:
                    check(a, atoms, format)
