from ase import Atoms
from ase.build import fcc111,add_adsorbate
from math import pi,ceil
from pandas import DataFrame

__all__ = ['Electrodes','zmatToAtoms','atomsToZmat']

class Electrodes:

    size=[4,4,2]
    distance=1.5
    atom='Au'
    position='hcp'
    offset=1
    atoms = Atoms()
    zmat = DataFrame()

    def __init__(self,mol):
        self.mol = mol

    def _attach(self):
        c = fcc111(self.atom,size=self.size)
        b = fcc111(self.atom,size=self.size)
        add_adsorbate(b,self.mol,self.distance,self.position,offset=self.offset)
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
