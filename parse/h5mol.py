from parse import xyz
import h5py
from util import *
from ase import Atom

class Parser(xyz.Parser):

    def _parsezmat(self):
        self.logger.debug('Building zmatrix...')
        mol = h5py.File(self.fn, 'r')
        zmat = ZMatrix()
        for i in range(0, len(mol['atomicNumbers'])):
            zmat += Atom( mol['atomicNumbers'][i], mol['coordinates'][i] )
        if self.opts.project:
            zmat.toZaxis()
        elif self.opts.sortaxis:
            zmat.sort(self.opts.sortaxis)
        self.zmat = zmat
        self.logger.info('Found: %s' % self.zmat.get_chemical_formula())
