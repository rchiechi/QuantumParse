import logging
from util import ZMatrix,elements
from ase import Atom
from cclib.io import ccread

__ALL__ = ['Parser']

class Parser:

    zmat = ZMatrix()
    ccparsed = None
    logger = logging.getLogger('Parser')
    lattice = {'constant':None,
               'vectors':[]}

    # When parsing large files set lowercase strings
    # that terminate the z-matrix to avoid looping
    # over the entire file
    breaks = ()
    # Optionally set a starting point to search for
    # coordinates (helpful with orca outputs)
    begin = ()

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
        self.logger.debug('Building zmatrix...')
        if self.opts.nocclib or not self.__cclibparse():
            self.__internalparse()
        if self.opts.project:
            self.zmat.toZaxis()
        elif self.opts.sortaxis:
            self.zmat.sort(self.opts.sortaxis)
        if self.opts.transport:
            self.__dotransport()
        if self.opts.build:
            if self.zmat.findElectrodes():
                self.logger.warn("This zmatrix already appears to have electrodes!")
            self.zmat.buildElectrodes(self.opts.build,self.opts.size,
                                      self.opts.distance,self.opts.binding,
                                      self.opts.surface,
                                      adatom=self.opts.adatom,
                                      SAM=self.opts.SAM,
                                      reverse=self.opts.reverse)
        self.zmat.findElectrodes()
        self.logger.info('Found: %s' % self.zmat.get_chemical_formula())

    def __internalparse(self):
        self.logger.warn('Could not parse with cclib, falling back to internal parser')
        zmat = ZMatrix()
        in_zmat = False
        if not self.begin or (self.fn[-4:].lower() in ('.com')):
            in_zmat = True
        with open(self.fn) as fh:
            for _l in fh:
                if not _l.strip():
                    continue
                if _l.strip() in self.begin:
                    self.logger.debug("Hit start in Z-matrix (%s)", _l.strip())
                    in_zmat = True
                    if len(zmat):
                        self.logger.warn("It looks like I am about to parse another z-matrices, dumping the last one!")
                        zmat = ZMatrix()
                if _l.strip() in self.breaks:
                    self.logger.debug("Hit break in Z-matrix (%s)", _l.strip())
                    in_zmat = False
                    continue
                row = []
                for _ls in _l.split():
                    if _ls.strip() and in_zmat:
                        row.append(_ls.strip())
                if not row or (row[0] not in elements):
                    continue
                if len(row) >= 4:
                    try:
                        x,y,z = map(float,row[1:4])
                        zmat += Atom(row[0],[x,y,z])
                    except ValueError:
                        self.logger.debug("Error parsing coordinates in Z-matrix in %s", self.fn)
                        self.logger.debug(' '.join(row))
                    except KeyError:
                        self.logger.debug("Error parsing atom name in Z-matrix in %s", self.fn)
                        self.logger.debug(' '.join(row))
            self.zmat = zmat

    def __cclibparse(self):
        self.logger.info("Using cclib to parse input; it may take a while...")
        zmat = ZMatrix()
        try:
            fh = ccread(self.fn)
        except NameError:
            return None
        except IndexError:
            return None
        except AttributeError:
            self.logger.warn("There is a bug in cclib preventing it from functionging with pybel.")
            return None
        if not fh:
            return None
        try:
            for i in range(0, len(fh.atomnos)):
                zmat += Atom(fh.atomnos[i], fh.atomcoords[-1][i])
        except AttributeError as msg:
            self.logger.error("Error parsing input %s" % str(msg))
            return None
#        if self.opts.project:
#            zmat.toZaxis()
#        elif self.opts.sortaxis:
#            zmat.sort(self.opts.sortaxis)
        self.zmat = zmat
        self.ccparsed = fh
        self.logger.info('Found: %s' % self.zmat.get_chemical_formula())
        if hasattr(fh, 'homos') and hasattr(fh, 'moenergies'):
            self.logger.info('HOMO/LUMO (eV): %0.4f/%0.4f' % (fh.moenergies[0][fh.homos[0]],
                                                              fh.moenergies[0][fh.homos[0]+1]))
        return True

    def __dotransport(self):
        self.logger.debug('Calling dummy __dotransport')
        return None
