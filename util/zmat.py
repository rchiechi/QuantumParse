from ase import Atom,Atoms
from ase.build import fcc111,add_adsorbate
from math import ceil,pi
from collections import Counter
import numpy as np
import logging
from .constants import *

class ZMatrix(Atoms):
    
    logger = logging.getLogger('Z-Matrix')
    electrodes = {'L':(0,0),'M':(0,0),'R':(0,0),'atom':None}

    def findElectrodes(self):
        '''Guess indices of electrode atoms.'''
        for e in EATOMS:
            if e in self.get_chemical_symbols():
                break
        self.logger.debug('Searching for %s electrodes.' % e)
        _l,_m,_r = [],[],[] 
        for _a in self:
            if _a.symbol == e:
                if not _m:
                    _l.append(_a.index)
                    self[_a.index].tag = TAGS['leadL']
                else:
                    _r.append(_a.index)
                    self[_a.index].tag = TAGS['leadR']
            else:
                _m.append(_a.index)
                self[_a.index].tag = TAGS['molecule']
        if not _m:
            self.logger.debug('This zmatrix looks like an electrode.')
            return
        elif not _l or not _r:
            self.logger.debug('No electrodes found.')
            return
        self.electrodes = {'L':(_l[0],_l[-1]),'M':(_m[0],_m[-1]),'R':(_r[0],_r[-1]),'atom':e}

    def buildElectrodes(self,atom,size,offset=1,distance=1.5,position='hcp'):
        '''Try to build electrodes around a molecule
           projected along the Z axis.'''
        if self.onAxis() != 'z':
            self.toZaxis()
            if self.onAxis() != 'z':
                self.logger.warn('Molecule is not projected along Z-axis!')
        self.logger.info('Building %s electrodes.' % atom)
        c = fcc111(atom,size=size)
        b = fcc111(atom,size=size)
        add_adsorbate(b,self,distance,position,offset=offset)
        b.rotate('x',pi)
        b.translate([0,0,ceil(abs(b[-1].z))])
        b.rotate('z',(4/3)*pi)
        add_adsorbate(c,b,1.5,'hcp',offset=1,mol_index=-1)
        self.__init__(c)
        self.sort()
        self.findElectrodes()
    
    def GetnpArray(self):
        a = []
        for _a in self:
            b = [0,0,0,0,0]
            b[0] = _a.x
            b[1] = _a.y
            b[2] = _a.z
            b[3] = _a.number
            b[4] = _a.symbol
            a.append(tuple(b)) #must be tuples for dt to work!

        dt = np.dtype([('x','float64'),('y','float64'),('z','float64'),
                       ('n','int8'),('atom',np.str_,2)])
        return np.array(a,dtype=dt)

    def sort(self,axis='z'):
        '''Sort zmatrix along given axis.'''
        self.logger.debug('Sorting along %s-axis' % axis)
        am = {'x':0,'y':1,'z':2}
        self.__init__(sorted(self, key=lambda self: self.position[am[axis]]))
        
    def write(self,fh):
        for _a in self:
            fh.write('%s\t%.8f\t%.8f\t%.8f\n' % (_a.symbol,_a.x,_a.y,_a.z) )

    def getAtomCounts(self):
        return Counter(self.get_chemical_symbols())
    def unique(self):
        return tuple(self.getAtomCounts().keys())

    def findDistances(self):
        distances = {}
        for _a in self:
            for _b in self:
                distances[self.get_distance(_a.index,_b.index)] = (_a.index,_b.index)
        return distances

    def onAxis(self):
        distances = self.findDistances()
        maxd = distances[ max(list(distances.keys())) ]
        f = np.array([self[maxd[0]].x,self[maxd[0]].y,self[maxd[0]].z])
        l = np.array([self[maxd[1]].x,self[maxd[1]].y,self[maxd[1]].z])
        diff = abs(f-l)
        if diff.max() == diff[0]:
            return 'x'
        elif diff.max() == diff[1]:
            return 'y'
        elif diff.max() == diff[2]:
            return 'z'
        else:
            return None

    def toZaxis(self):
        axis = self.onAxis()
        if axis == 'z':
            self.logger.debug('Already on z-axis, skipping rotation.')
            return
        self.logger.debug('Rotating from %s to z-axis.' % axis)
        if axis == 'x':
            self.rotate('y',pi/2)
        elif axis == 'y':
            self.rotate('x',pi/2)
        zc = {}
        for _a in self:
            zc[_a.z] = [_a.x,_a.y]
        zmin = min(list(zc.keys()))
        xmin,ymin = zc[zmin]
        self.translate([-1*xmin,-1*ymin,-1*zmin])
        self.sort()

