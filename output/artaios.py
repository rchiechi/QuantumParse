#!/usr/bin/env python3

import sys,os
from parse.orca import *

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
	else:
		print("I need an input file!")
		sys.exit()
	if not os.path.isfile(fn):
		print("I can't read %s" % fn)
		sys.exit()
	with open(fn, 'rt') as fh:
		orbs,orbidx = norbs(fh)
		fm = fock(fh)
		ol = overlap(fh)
	nb = fm.shape[0]
	countiorb(orbs,nb)
	with open('hamiltonian.1', 'wt') as fh:
		l = 0
		for i in range(0,nb):
			for n in range(0,nb):
				fh.write("%s\t" % fm[i][n])
				l += 1
				if l >= 4:
					fh.write("\n")
					l = 0
	with open('overlap', 'wt') as fh:
		l = 0
		for i in range(0,nb):
			for n in range(0,nb):
				fh.write("%s " % ol[i][n])
				l += 1
				if l >= 4:
					fh.write("\n")
					l = 0

	guessorbs = {"L":[0,0], "M":[0,0], "R":[0,0]}
	with open('transport.in', 'tw') as fh:
		fh.write("#Here is a list of orbitals on each atom:\n")
		i=1
		for a in orbidx:
			ie = i+len(orbs[a])-1
			fh.write("#%s %s-%s\n" % (a, i, ie) )
			if 'Au' in a:
				if guessorbs["L"][0] == 0:
					guessorbs["L"][0] = i
				elif guessorbs["M"] == [0,0]:
					guessorbs["L"][1] = ie
				elif guessorbs["R"][0] == 0:
					guessorbs["R"][0] = i
				else:
					guessorbs["R"][1] = ie
			elif guessorbs["M"][0] == 0:
					guessorbs["M"][0] = i
			else:
				guessorbs["M"][1] = ie
			i += len(orbs[a])

		fh.write("##############\n")
		wrblock(fh,["$partitioning","   totnbas  %s" % nb, "   leftbas %s-%s #EDIT THIS!" % tuple(guessorbs["L"]),\
				"   centralbas %s-%s #EDIT THIS!" % tuple(guessorbs["M"]), \
				"   rightbas %s-%s #EDIT THIS!" % tuple(guessorbs["R"]),"$end"])
		wrblock(fh,["$energy_range","  units   eV","  start  -8.0","   end     -1.0",\
				"   fermi_level -5.0","   steps 500", "$end"])
		wrblock(fh,["$system","   nspin  1","$end"])
		wrblock(fh,["$electrodes", "   self_energy wbl","   dos_s 0.036","$end"])
		fh.write("ham_conv 27.21\n")
		fh.write("modelham T\n")
		fh.write("qcprog gen\n")
