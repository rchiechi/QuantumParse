from ase import Atoms
from ase.build import fcc111,add_adsorbate
from math import pi,ceil
from pandas import DataFrame

__all__ = ['buildElectrodes','zmatToAtoms','atomsToZmat','findDistances']

def buildElectrodes(atoms,atom='Au',size=[4,4,2],position='hcp',distance=1.5,offset=1):

    c = fcc111(atom,size=size)
    b = fcc111(atom,size=size)
    add_adsorbate(b,atoms,distance,position,offset=offset)
    b.rotate('x',pi)
    b.translate([0,0,ceil(abs(b[-1].z))])
    b.rotate('z',(4/3)*pi)
    add_adsorbate(c,b,1.5,'hcp',offset=1,mol_index=len(b)-1)
    return c

def zmatToAtoms(zmat):
    atoms = Atoms()
    for a, group in zmat.groupby('atoms'):
        f = ''
        pos = []
        if a in ('Au','Ag'):
            tag = 1
        elif a in ('S'):
            tag = 2
        else:
            tag = 0
        for row in group.iterrows():
            f+=a
            pos.append((row[1]['x'],row[1]['y'],row[1]['z']))
        atoms += Atoms(f,positions=pos,tags=tag)
    return atoms

def atomsToZmat(atoms):
    zmat = {'atoms':[],'x':[],'y':[],'z':[]}
    for a in atoms:
        zmat['atoms'].append(a.symbol)
        zmat['x'].append(a.x)
        zmat['y'].append(a.y)
        zmat['z'].append(a.z)
    return DataFrame(zmat)

def findDistances(xyz):
    distances = {}
    if type(xyz) == type(Atoms()):
        for _a in xyz:
            for _b in xyz:
                c,d = np.array([_a.x,_a.y,_a.z]), np.array([_b.x,_b.y,_b.z])
                distances[np.linalg.norm(c-d)] = (_a.index,_b.index)
    elif type(xyz) == type(DataFrame()):
        for _a in xyz.iterrows():
            for _b in xyz.iterrows():
                c,d = np.array([_a[1].x,_a[1].y[1],_a.z]), np.array([_b[1].x,_b[1].y,_b[1].z])
                distances[np.linalg.norm(c-d)] = (_a[0],_b[0])
    return distances
