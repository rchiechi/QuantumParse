#!/usr/bin/env python3

#
# THIS IS A TEST: DO NOT USE
#

import sys,os
from ase.io import read,write
from ase.build import fcc111,add_adsorbate

height=1.5
position='hcp'
size=[10,10,2]

slab=fcc111('Au',size)
slab.info['adsorbate_info']['top layer atom index'] = len(b.positions)-1
AC=read('AQ.xyz')
print(AC)
for i in range(1,size[0]-1,2):
    add_adsorbate(slab,AC,height,position,offset=[0,i],mol_index=0)
    for n in range(2,size[0]-1,2):
        add_adsorbate(slab,AC,height,position,offset=[n,i],mol_index=0)

    

write('test.png',b,rotation='80x,180z')
write('test.xyz',b)
def add_mol(slab,ads):
    # Get the z-coordinate:
    if 'top layer atom index' in info:
        a = info['top layer atom index']
    else:
        a = slab.positions[:, 2].argmax()
        if 'adsorbate_info' not in slab.info:
            slab.info['adsorbate_info'] = {}
        slab.info['adsorbate_info']['top layer atom index'] = a
    z = slab.positions[a, 2] + height

    # Move adsorbate into position
    ads.translate([pos[0], pos[1], z] - ads.positions[mol_index])

    # Attach the adsorbate
    slab.extend(ads)
