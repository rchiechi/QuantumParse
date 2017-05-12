#!/usr/bin/env python3

#
# THIS IS A TEST: DO NOT USE
#

import sys,os,argparse
from ase.io import read,write
from ase.build import fcc111,add_adsorbate



# Parse command line arguments
desc='Build a SAM from an XYZ file containing a molecule'
parser = argparse.ArgumentParser(description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infile', type=str, nargs=1, default=[], 
    help='XYZ file to parse.')
parser.add_argument('-H', '--height', type=int, default=1.5, 
    help='Height in Angstroms to place molecules above surface.')
parser.add_argument('-t', '--tilt', type=int, default=0, 
    help='Tilt angle.')
parser.add_argument('-A', '--tiltaxis', type=str, default='x', 
    help='Tilt axis.')
parser.add_argument('-e','--electrode', type=str, default='Au', 
    help='Type of electrode.')
parser.add_argument('-s','--size', type=str, default='10,10,2', 
    help='Size of the substrate in the form x,y,z number of atoms.')


opts=parser.parse_args()

# Needed for zmatrix parser class
opts.build = False
opts.project = True
from parse.xyz import Parser

if not opts.infile:
    print("I need an input file.")
    sys.exit()

position='hcp'
try:
    opts.size = tuple(map(int, opts.size.strip().split(',')))
except:
    print("Error parsing size as three comma-separated integers")
    sys.exit()

slab=fcc111(opts.electrode, opts.size)
# Needed to prevent multilayer formation
slab.info['adsorbate_info']['top layer atom index'] = len(slab.positions)-1
xyz = Parser(opts,opts.infile[0])
xyz.parseZmatrix()

if opts.tilt > 0:
    xyz.zmat.rotateAboutAxis(opts.tiltaxis,opts.tilt)

mol = xyz.getZmat()
print(mol)
for i in range(1,opts.size[0]-1,2):
    add_adsorbate(slab,mol,opts.height,position,offset=[0,i],mol_index=0)
    for n in range(2,opts.size[0]-1,2):
        add_adsorbate(slab,mol,opts.height,position,offset=[n,i],mol_index=0)
write(opts.infile[0][:-4]+'_SAM.png',slab,rotation='80x,180z')
write(opts.infile[0][:-4]+'_SAM.xyz',slab)
