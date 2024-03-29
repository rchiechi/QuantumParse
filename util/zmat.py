from ase import Atom,Atoms
import ase.build
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from math import ceil
from collections import Counter
import numpy as np
import logging
from .constants import EATOMS,TAGS

__ALL__ = ['ZMatrix']

class ZMatrix(Atoms):

    logger = logging.getLogger('Z-Matrix')
    electrodes = {'L':(0,0),'M':(0,0),'R':(0,0),'atom':None}
    optimized = None

    def findElectrodes(self):
        '''Guess indices of electrode atoms.'''
        for e in EATOMS:
            if e in self.get_chemical_symbols():
                break
        self.logger.debug('Searching for %s electrodes.', e)
        _l,_m,_r = [],[],[]
        for _a in self:
            self.logger.debug(str(_a))
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
            if _l or _r:
                self.logger.debug('This zmatrix looks like an electrode.')
                self.logger.debug(_l)
                self.logger.debug(_r)
            else:
                self.logger.debug('No electrodes found.')
            return None
        if not _l or not _r:
            self.logger.debug('No electrodes found.')
            return False
        self.electrodes = {'L':(_l[0],_l[-1]),'M':(_m[0],_m[-1]),'R':(_r[0],_r[-1]),'atom':e}
        self.logger.debug(self.electrodes)
        return True

    def buildElectrodes(self,atom,size,distance,position,surface,**kwargs):
        '''Try to build electrodes around a molecule
           projected along the Z axis.'''

        for _kw in ('adatom', 'SAM', 'reverse'):
            if _kw not in kwargs:
                kwargs[_kw] = False
        if 'anchor' not in kwargs:
            kwargs['anchor'] = 'S'

        if 'z' not in self.toZaxis(kwargs['reverse']):
            self.logger.warning('Molecule is not projected along Z-axis!')
        # if self[0].symbol != 'S' or self[-1].symbol != 'S':
        if self[0].symbol != kwargs['anchor'] and self[-1].symbol != kwargs['anchor']:
            self.logger.warning(
                'Molecule is not terminated with at least one %s atom!', kwargs['anchor'])
        self.logger.info('Building %s electrodes.', atom)
        b = getattr(ase.build,surface)(atom,size=size)
        c = getattr(ase.build,surface)(atom,size=size)
        if kwargs['adatom']:
            self.logger.debug('Adding %s adatom', atom)
            Spos = self[-1].position
            self += Atom(atom,position=[Spos[0],Spos[1],Spos[2]+2.5])
        if kwargs['SAM']:
            offset = 0
            self.logger.debug('Building an n x n SAM (%s)', str(size[0]/2))
            for i in range(0,size[0],2):
                ase.build.add_adsorbate(b,self,distance,position,offset=[0,i])
                for j in range(2,size[0],2):
                    ase.build.add_adsorbate(b,self,distance,position,offset=[j,i])
        else:
            offset = (ceil(size[0]/2-1), ceil(size[1]/2-1))
            ase.build.add_adsorbate(b,self,distance,position,offset=offset)
        self.logger.debug('Electrode size: %s offset: %s distance:%s',
                          str(size),str(offset),str(distance))
        # b.rotate('x',pi)
        b.rotate(180,'x')
        b.translate([0,0,ceil(abs(b[-1].z))])
        # b.rotate('z',(4/3)*pi)
        b.rotate(240,'z')
        ase.build.add_adsorbate(c,b,distance,position,offset=offset,mol_index=-1)
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
            a.append(tuple(b))  # NOTE: must be tuples for dt to work!

        dt = np.dtype([('x','float64'),('y','float64'),('z','float64'),
                       ('n','int8'),('atom',np.str_,2)])
        return np.array(a,dtype=dt)

    def sort(self,axis='z'):
        '''Sort zmatrix along given axis.'''
        self.logger.info('Sorting along %s-axis', axis)
        am = {'x':0,'y':1,'z':2}
        self.__init__(sorted(self, key=lambda self: self.position[am[axis]]))
        for _a in self:
            self.logger.debug(str(_a))

    def write(self,fh):
        for _a in self:
            fh.write('%s\t%.8f\t%.8f\t%.8f\n' % (_a.symbol,_a.x,_a.y,_a.z))

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

    def optimize(self):
        self.calc = EMT()
        dyn = BFGS(self)
        dyn.run(fmax=0.05)
        self.optimized = ZMatrix()
        for _atom in dyn.atoms:
            self.optimized += _atom

    def onAxis(self):
        distances = self.findDistances()
        maxd = distances[max(list(distances.keys()))]
        _f = np.array([self[maxd[0]].x,self[maxd[0]].y,self[maxd[0]].z])
        _l = np.array([self[maxd[1]].x,self[maxd[1]].y,self[maxd[1]].z])
        diff = abs(_f-_l)
        if diff.max() == diff[0]:
            return 'x'
        elif diff.max() == diff[1]:
            return 'y'
        elif diff.max() == diff[2]:
            if self[-1].position[2] - self[0].position[2] > 0:
                return 'z'
            else:
                return '-z'
        else:
            self.logger.debug('Error determining projection axis.')
            return None

    def toZaxis(self, reverse=False):
        if reverse:
            return self.toAxis('-z')
        else:
            return self.toAxis('z')

    def toAxis(self, target_axis):
        axis = self.onAxis()
        if axis == target_axis:
            self.logger.debug('Already on %s-axis, skipping rotation.', target_axis)
            return axis
        self.logger.debug('Rotating from %s-axis to %s-axis.', axis, target_axis)
        self.rotate(axis, target_axis)
        # if axis == 'x':
        #    self.rotate('x','z')
        # elif axis == 'y':
        #    self.rotate('x',pi/2)
        self.__moveAfterRotate()
        return self.onAxis()

    def __moveAfterRotate(self):
        zc = {}
        for _a in self:
            zc[_a.z] = [_a.x,_a.y]
        zmin = min(list(zc.keys()))
        xmin,ymin = zc[zmin]
        self.translate([-1*xmin,-1*ymin,-1*zmin])
        self.sort()

    def rotateAboutAxis(self,axis,degree):
        '''Project molecule along Z-axis and rotate around another axis'''
        self.toZaxis()
        self.logger.debug('Rotating around %s by %s°.' % (axis,degree))
        self.rotate(float(degree),axis)
        self.__moveAfterRotate()
