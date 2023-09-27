#!/usr/bin/env python3

import argparse


# Parse command line arguments
desc = 'Perform some manipulations on ams files'

parser = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infiles', type=str, nargs='*', default=[],
                    help='Ams file to parse.')
parser.add_argument('-P', '--eplot', action='store_true', default=False,
                    help='Planarize a structure.')

opts = parser.parse_args()


class AMSmatrix:
    
    def __init__(self, fn):
        self.zmat = self._parseams(fn)
    
    def _parseams(self, fn):
        _lines = []
        with open(fn, 'rt') as fh:
            for _l in fh:
                _lines.append(_l.strip())
        in_atoms = False
        natoms = 0
        atoms = {}
        for i, _l in enumerate(_lines):
            if _l == 'Number of atoms':
                natoms = int(_lines[i+1].split(' ')[1])
            if _l == 'Atom data':
                in_atoms = True
                continue
            if _l == 'DUMMYELEMENT':
                in_atoms = False
                continue
            if in_atoms:
                _atom = _l.split(',')
                _idx = int(_atom[1])
                if _idx not in atoms:
                    _atoms[_idx] = []
        print(f'Found {natoms} atoms in {fn}')
        
zmats = []
for in_file in opts.infiles:
    zmats.append(AMSmatrix(in_file)) 
