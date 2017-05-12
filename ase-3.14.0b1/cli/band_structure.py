from __future__ import print_function

import numpy as np

from ase.io import read
from ase.dft.kpoints import (get_monkhorst_pack_size_and_offset,
                             monkhorst_pack_interpolate,
                             bandpath)
from ase.dft.band_structure import BandStructure


class CLICommand:
    short_description = 'Plot band-structure'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('calculation')
        parser.add_argument('-q', '--quiet', action='store_true')
        parser.add_argument('-k', '--path', help='Example "GXL".')
        parser.add_argument('-n', '--points', type=int, default=50,
                            help='Number of point along the path '
                            '(default: 50)')
        parser.add_argument('-r', '--range', default='-10,5',
                            help='Default value: "-10,5" '
                            '(in eV relative to Fermi level).')

    @staticmethod
    def run(args, parser):
        main(args, parser)


def main(args, parser):
    atoms = read(args.calculation)
    calc = atoms.calc
    bzkpts = calc.get_bz_k_points()
    ibzkpts = calc.get_ibz_k_points()
    efermi = calc.get_fermi_level()
    nibz = len(ibzkpts)
    nspins = 1 + int(calc.get_spin_polarized())
    eps = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                     for k in range(nibz)]
                    for s in range(nspins)])
    if not args.quiet:
        print('Spins, k-points, bands: {}, {}, {}'.format(*eps.shape))
    try:
        size, offset = get_monkhorst_pack_size_and_offset(bzkpts)
    except ValueError:
        path = ibzkpts
    else:
        if not args.quiet:
            print('Interpolating from Monkhorst-Pack grid (size, offset):')
            print(size, offset)
        if args.path is None:
            parser.error('Please specify a path!')
        bz2ibz = calc.get_bz_to_ibz_map()
        path = bandpath(args.path, atoms.cell, args.points)[0]
        icell = atoms.get_reciprocal_cell()
        eps = monkhorst_pack_interpolate(path, eps.transpose(1, 0, 2),
                                         icell, bz2ibz, size, offset)
        eps = eps.transpose(1, 0, 2)

    emin, emax = (float(e) for e in args.range.split(','))
    bs = BandStructure(atoms.cell, path, eps, reference=efermi)
    bs.plot(emin=emin, emax=emax)
