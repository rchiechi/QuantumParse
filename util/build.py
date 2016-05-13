from ase import Atoms
from ase.build import fcc111,add_adsorbate
from math import pi,ceil
from pandas import DataFrame

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
    
