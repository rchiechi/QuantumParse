from parse import xyz
import logging
from util import *
from ase import Atom,Atoms
from cclib.io import ccread

class Parser(xyz.Parser):
    breaks = ('--link1--','natoms=','stoichiometry')

    def _parsezmat(self):
        self.logger.debug('Building zmatrix...')
        zmat = ZMatrix()
        fh = ccread(self.fn)
        for i in range(0, len(fh.atomnos)):
            zmat += Atom(fh.atomnos[i], fh.atomcoords[-1][i])
        if self.opts.project:
            zmat.toZaxis()
        elif self.opts.sortaxis:
            zmat.sort(self.opts.sortaxis)
        self.zmat = zmat
        self.logger.info('Found: %s' % self.zmat.get_chemical_formula())
