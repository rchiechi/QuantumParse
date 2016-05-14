from ase import Atoms
from ase.build import fcc111,add_adsorbate
from math import pi,ceil
from pandas import DataFrame
import logging
import numpy as np

__all__ = ['buildElectrodes','zmatToAtoms','atomsToZmat','sortZmat','findDistances','onAxis','toZaxis']

logger = logging.getLogger('Builder')

def buildElectrodes(atoms,atom='Au',size=[4,4,2],position='hcp',distance=1.5,offset=1):
    logger.info('Building %s electrodes.' % atom)
    c = fcc111(atom,size=size)
    b = fcc111(atom,size=size)
    add_adsorbate(b,atoms,distance,position,offset=offset)
    b.rotate('x',pi)
    b.translate([0,0,ceil(abs(b[-1].z))])
    b.rotate('z',(4/3)*pi)
    add_adsorbate(c,b,1.5,'hcp',offset=1,mol_index=-1)
    return c

def zmatToAtoms(zmat):
    f = ''
    pos = []
    tag = []
    for row in zmat.iterrows():
        a = row[1]['atoms']
        if a in ('Au','Ag'):
            tag.append(1)
        elif a in ('S'):
            tag.append(2)
        else:
            tag.append(0)
        f+=a
        pos.append((row[1]['x'],row[1]['y'],row[1]['z']))
    atoms = Atoms(f,positions=pos,tags=tag)
    return atoms

def atomsToZmat(atoms):
    zmat = {'atoms':[],'x':[],'y':[],'z':[]}
    for a in atoms:
        zmat['atoms'].append(a.symbol)
        zmat['x'].append(a.x)
        zmat['y'].append(a.y)
        zmat['z'].append(a.z)
    return DataFrame(zmat)

def sortZmat(zmat,axis='z'):
    idx = []
    for i in range(0,len(zmat['atoms'])):
        idx.append(i)
    zmat = zmat.sort_values(axis)
    zmat.index = idx
    return zmat

def findDistances(xyz):
    distances = {}
    if type(xyz) == type(Atoms()):
        for _a in xyz:
            for _b in xyz:
                distances[xyz.get_distance(_a.index,_b.index)] = (_a.index,_b.index)
    elif type(xyz) == type(DataFrame()):
        for _a in xyz.iterrows():
            for _b in xyz.iterrows():
                c,d = np.array([_a[1].x,_a[1].y[1],_a.z]), np.array([_b[1].x,_b[1].y,_b[1].z])
                distances[np.linalg.norm(c-d)] = (_a[0],_b[0])
    return distances

def onAxis(xyz):
    distances = findDistances(xyz)
    maxd = distances[ max(list(distances.keys())) ]
    f = np.array([xyz[maxd[0]].x,xyz[maxd[0]].y,xyz[maxd[0]].z])
    l = np.array([xyz[maxd[1]].x,xyz[maxd[1]].y,xyz[maxd[1]].z])
    diff = abs(f-l)
    if diff.max() == diff[0]:
        return 'x'
    elif diff.max() == diff[1]:
        return 'y'
    elif diff.max() == diff[2]:
        return 'z'
    else:
        return None

def toZaxis(atoms):
    logger.debug('Rotating zmatrix to Z-axis.')
    axis = onAxis(atoms)
    if axis == 'x':
        atoms.rotate('y',pi/2)
    elif axis == 'y':
        atoms.rotate('x',pi/2)
    zc = []
    for _a in atoms:
        zc.append(_a.z)
    diff = min(zc)
    atoms.translate([0,0,-1*diff])
    zmat = atomsToZmat(atoms)
    zmat = sortZmat(zmat)
    atoms = zmatToAtoms(zmat)
    #TODO reindex so min atom == min index
    return atoms,zmat
