#!/usr/bin/env python3

import sys,os
from parse import *



def wrblock(fh,l):
	l[-1] = l[-1]+"\n"
	fh.write("\n".join(l))

def countiorb(orbs,nb):
	norb = 0
	for a in orbs:
		norb+= len(orbs[a])
	print("Total iorbs: %s" % norb)
	if nb != norb:
		print("Error: mismatch in number of orbtials and fock matrix")

if __name__ == "__main__":
	if len(sys.argv) > 1:
		fn = sys.argv[1]
	if not os.path.isfile(fn):
		print("I can't read %s" % fn)
		sys.exit()
	with open(fn, 'rt') as fh:
		orbs,orbidx = norbs(fh)
		fm = fock(fh)
		ol = overlap(fh)
	nb = fm.shape[0]
	countiorb(orbs,nb)
	with open('Extended_Molecule', 'wt') as fh:
		wrblock(fh,["# name: nspin","# type: scalar"," 1"])
		wrblock(fh,["# name: FermiE","# type: scalar"," -4.5"])
		wrblock(fh,["# name: iorb","# type: matrix","# rows: %s" % nb,"# columns: 4"])
		i=0
		for a in sorted(orbs.keys()):
			i+=1
			for n in range(0,len(orbs[a])):
				fh.write("\t".join([str(i),'0','1','0'])+"\n")
		wrblock(fh,["# name: kpoints_EM","# type: matrix","# rows: 1","# columns: 3"])
		fh.write(" 0.0000000000E+00   0.0000000000E+00   1.0000000000E+00\n")
		wrblock(fh,["# name: HSM","# type: matrix","# rows: %s" % nb**2,"# columns: 7"])
		for i in range(0,nb):
			for n in range(0,nb):
				fh.write("     1    %s    %s   %.10E   0.0000000000E+00  %.10E   0.0000000000E+00\n" % (n+1,i+1,ol[i][n],fm[i][n]))
