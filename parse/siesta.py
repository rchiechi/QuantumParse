from parse import xyz
from util import elements
import re

class Parser(xyz.Parser):
    
    lattice = {'constant':None,
            'vectors':[]}

    def getLattice(self):
        return self.lattice

    def hasLattice(self):
        if self.lattice['constant'] == None or not self.lattice['vectors']:
            return False
        else:
            return True

    def _parsezmat(self):
        zmat = {'atoms':[],'x':[],'y':[],'z':[]}
        with open(self.fn) as fh:
            atomlabels = self._atomlabels(fh)
            inblock = False
            for l in fh:
                if '%block atomiccoordinatesandatomicspecies' in l.lower():
                    inblock = True
                    continue
                elif '%endblock atomiccoordinatesandatomicspecies' in l.lower():
                    break
                if inblock:
                    row = list(filter(None,re.split(self.ws,l)))
                    try:
                        x,y,z = map(float,row[:3])
                        i = int(row[3])
                        zmat['x'].append(x)
                        zmat['y'].append(y)
                        zmat['z'].append(z)
                        zmat['atoms'].append(atomlabels[i])
                    except ValueError as msg:
                        self.logger.debug("Error parsing line in Z-matrix in %s" % self.fn)
                        self.logger.debug(' '.join(row))
            self._zmattodf(zmat)
            self.parseLattice(fh)

    def parseLattice(self,fh):
        inblock = False
        fh.seek(0)
        for l in fh:
            if 'latticeconstant' in l.lower():
                self.lattice['constant'] = l.strip()
            if '%block latticevectors' in l.lower():
                inblock = True
                continue
            elif '%endblock latticevectors' in l.lower():
                break
            if inblock:
                self.lattice['vectors'].append(l.strip())
        fh.seek(0)

    def _atomlabels(self,fh):
        atomlabels = {}
        inblock = False
        fh.seek(0)
        for l in fh:
            if '%block chemicalspecieslabel' in l.lower():
                inblock = True
                continue
            elif '%endblock chemicalspecieslabel' in l.lower():
                break
            if inblock:
                i,n,atom = list(filter(None,re.split(self.ws,l)))[:3]
                atomlabels[int(i)] = atom
                self.logger.debug('Found atom %s' % atom)
        fh.seek(0)
        return atomlabels
