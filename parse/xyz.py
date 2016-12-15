import logging
from util import *
from ase import Atom,Atoms

class Parser:
  
    zmat = ZMatrix()
    logger = logging.getLogger('Parser')
    lattice = {'constant':None,
            'vectors':[]}

    #When parsing large files set lowercase strings
    #that terminate the z-matrix to avoid looping
    #over the entire file
    breaks = ()

    def __init__(self,opts,fn):
        self.fn = fn
        self.opts = opts

    def getLattice(self):
        return self.lattice

    def hasLattice(self):
        return False

    def setLattice(self,lattice):
        self.lattice = lattice

    def haselectrodes(self):
        for a in ('L','M','R'):
            if self.zmat.electrodes[a] == (0,0) or -1 in self.zmat.electrodes[a]:
                return False
        return True

    def setZmat(self,zmat):
        self.zmat = zmat
    
    def setAtoms(self,atoms):
        self.atoms = atoms

    def getZmat(self):
        return self.zmat
    def getAtoms(self):
        return self.atoms

    def parseZmatrix(self):
        self._parsezmat()
        if self.opts.build:
            self.zmat.buildElectrodes(self.opts.build,self.opts.size,
                    self.opts.distance,self.opts.binding,self.opts.surface,self.opts.adatom,self.opts.SAM)
        self.zmat.findElectrodes()

    def _parsezmat(self):
        self.logger.debug('Building zmatrix...')
        zmat = ZMatrix()
        with open(self.fn) as fh:
            for l in fh:
                row = []
                for _l in l.split():
                    if _l.strip(): row.append(_l.strip())
                if not row: 
                    continue
                elif row[0].lower() in self.breaks: 
                    self.logger.debug("Hit break in Z-matrix (%s)" % l.strip())
                    break
                elif row[0] not in elements:
                    continue
                elif row[0] not in elements and len(zmat):
                    self.logger.warn("It looks like I am about to parse two z-matrices, breaking!")
                    break
                if len(row) >= 4:
                    try:
                        x,y,z = map(float,row[1:4])
                        zmat += Atom(row[0],[x,y,z])
                    except ValueError:
                        self.logger.debug("Error parsing coordinates in Z-matrix in %s" % self.fn)
                        self.logger.debug(' '.join(row))
                    except KeyError:
                        self.logger.debug("Error parsing atom name in Z-matrix in %s" % self.fn)
                        self.logger.debug(' '.join(row))
        if self.opts.project:
            zmat.toZaxis()
        elif self.opts.sortaxis:
            zmat.sort(self.opts.sortaxis)
        self.zmat = zmat
    
