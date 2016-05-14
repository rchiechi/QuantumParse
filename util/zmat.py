from ase import Atoms
from ase.build import fcc111,add_adsorbate
from math import ceil,pi
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
                self.logger.debug('Guessing %s electrodes.' % e)
                break
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

    def buildElectrodes(self,atom,size=[4,4,2],offset=1,distance=1.5,position='hcp'):
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
        #for _a in self:
        #    print(_a)

    def GetnpArray(self,axismap):
        a = []
        for _a in self:
            b = [0,0,0,0]
            b[axismap['i']] = _a.index
            b[axismap['x']] = _a.x
            b[axismap['y']] = _a.y
            b[axismap['z']] = _a.z
            a.append(b)
        return np.array(a)

    def sort(self,axis='z'):
        '''Sort zmatrix along given axis.'''
        ox = {'x':0,'y':1,'z':2}
        del(ox[axis])
        ox = list(ox.keys())
        axismap = {axis:0,ox[0]:1,ox[1]:2,'i':3}
        a = self.GetnpArray(axismap)
        f = ''
        pos = []
        #print(a)
        for ax in np.sort(a,axis=0):
            _a = self[int(ax[axismap['i']])]
            f += _a.symbol
            pos.append( [_a.x,_a.y,_a.z] )
            #print(_a.symbol)
            #print(ax)
            #f += ax[axismap['a']]
            #pos.append( [ ax[axismap['x']],
            #              ax[axismap['y']],
            #              ax[axismap['z']] ])
        self.__init__(f,pos)
            
    def write(self,fh):
        for _a in self:
            fh.write('%s\t%s\t%s\t%s\n' % (_a.symbol,_a.x,_a.y,_a.z) )

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
        self.logger.debug('Rotating zmatrix to Z-axis.')
        axis = self.onAxis()
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

if __name__ == '__main__':
    d = 1.1
    zmat = ZMatrix('OC', positions=[(0, 0, 10), (0, 0, d)])
    zmat.sort()
    zmat.findElectrodes()
    zmat.toZaxis()
    for _a in zmat:
        print(_a)
