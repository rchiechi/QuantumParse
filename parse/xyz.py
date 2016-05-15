import logging
from util import *
from util.build import *
from time import sleep
import re
from ase import Atoms



class Parser:
  
    atoms = Atoms()
    zmat = ZMatrix()
    logger = logging.getLogger('Parser')
 
    lattice = {'constant':None,
            'vectors':[]}

    #When parsing large files set lowercase strings
    #that terminate the z-matrix to avoid looping
    #over the entire file
    breaks = ()
    ws = re.compile('\s+')

    def __init__(self,opts,fn):
        self.fn = fn
        self.opts = opts

    def getLattice(self):
        return self.lattice

    def hasLattice(self):
        return False

    def haselectrodes(self):
        for a in ('L','M','R'):
            if self.zmat.electrodes[a] == (0,0) or -1 in self.zmat.electrodes[a]:
                return False
        return True

    def setZmat(self,zmat):
        self.zmat = zmat
        #self.atoms = zmatToAtoms(zmat)

    def setAtoms(self,atoms):
        self.atoms = atoms
        #self.zmat = atomsTozmat(atoms)

    def getZmat(self):
        return self.zmat
    def getAtoms(self):
        return self.atoms

    def _parsezmat(self):
        zmat = {'atoms':[],'x':[],'y':[],'z':[]}
        f = []
        pos = []
        with open(self.fn) as fh:
            for l in fh:
                row = []
                for _l in re.split(self.ws,l):
                    if _l.strip(): row.append(_l.strip())
                if not row: continue
                elif row[0].lower() in self.breaks: 
                    self.logger.debug("Hit break in Z-matrix (%s)" % l.strip())
                    break
                elif row[0] not in elements:continue
                if len(row) >= 4:
                    try:
                        x,y,z = map(float,row[1:4])
                        zmat['x'].append(x)
                        zmat['y'].append(y)
                        zmat['z'].append(z)
                        zmat['atoms'].append(str(row[0]))
                        f.append(row[0].lower().capitalize())
                        pos.append((x,y,z))
                    except ValueError:
                        self.logger.debug("Error parsing line in Z-matrix in %s" % self.fn)
                        self.logger.debug(' '.join(row))
        self.zmat = ZMatrix(f,pos)
        if self.opts.sortaxis:
            self.zmat.sort(self.opts.sortaxis)

    def _zmattodf(self,zmat):
        if self.opts.sortaxis:
            self.logger.debug('Sorting Z-matrix by column %s' % self.opts.sortaxis)
            self.zmat = sortZmat(pd.DataFrame(zmat),self.opts.sortaxis)
        else:
            self.zmat = pd.DataFrame(zmat)
        if not len(self.zmat):
            self.logger.error('Empty Z-matrix parsed from %s' % self.fn)
            import sys
            sys.exit()
        self.atoms = zmatToAtoms(self.zmat)
        self.logger.info('Parsed a Z-matrix with %s atoms.' % len(self.zmat))
     

    def parseZmatrix(self):
        self._parsezmat()
        if self.opts.build:
            self.zmat.buildElectrodes(self.opts.build)
        self.zmat.findElectrodes()
         
