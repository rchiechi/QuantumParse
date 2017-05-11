#!/usr/bin/env python3

#
# THIS IS A TEST: DO NOT USE
#

import sys,os,argparse
from ase.io import read,write
from ase.build import fcc111,add_adsorbate



# Parse command line arguments
desc='Build a SAM from an XYZ file containing a molecule projected along the Z-axis'
parser = argparse.ArgumentParser(description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infile', type=str, nargs=1, default=[], 
    help='XYZ file to parse.')
parser.add_argument('-H', '--height', type=int, default=1.5, 
    help='Height in Angstroms to place molecules above surface.')
parser.add_argument('-e','--electrode', type=str, default='Au', 
    help='Type of electrode.')
parser.add_argument('-s','--size', type=str, default='10,10,2', 
    help='Size of the substrate in the form x,y,z number of atoms.')


opts=parser.parse_args()

if not infile:
    print("I need an input file.")
    sys.exit()

#height=1.5
position='hcp'
#size=[10,10,2]

opts.size = opts.size.strip().split(',')

slab=fcc111(opts.electrode,opts.size)
slab.info['adsorbate_info']['top layer atom index'] = len(slab.positions)-1
AC=read(opts.infile)
print(AC)
for i in range(1,size[0]-1,2):
    add_adsorbate(slab,AC,height,position,offset=[0,i],mol_index=0)
    for n in range(2,size[0]-1,2):
        add_adsorbate(slab,AC,height,position,offset=[n,i],mol_index=0)
write(opts.infile[:-4]+'.png',slab,rotation='80x,180z')
write(opts.infile[:-4]+'_SAM.xyz',slab)
