#!/usr/bin/env python3

import sys,os,re
import numpy as np

YELLOW="\033[1;33m"
WHITE="\033[0m"
RED="\033[1;31m"
TEAL="\033[1;36m"
GREEN="\033[1;32m"
BLUE="\033[1;34m"
RS="\033[0m"
CL="\033[2K"
UP="\033[1A"
CS="\033[0J"

SPINNER=('\\','|','/','|')
def overlap(fh):
	print("Parsing overlap matrix...")
	fh.seek(0)
	ovdict = {}
	inoverlap = False
	lk = []
	rp = re.compile('\s')
	for l in fh:
		lk.append(l.strip())
		if not inoverlap:
			if len(lk) > 3:
				lk.pop(0)
			else:
				continue
		if lk[1] == 'OVERLAP MATRIX' and not inoverlap:
			inoverlap = True
			continue
		elif l[0] == '-' and inoverlap:
			inoverlap = False
			break
		if inoverlap:
			lsf = filter(None, re.split(rp,l))
			fl = []
			for n in lsf:
				if '.' in n:
					fl.append(float(n))
				else:
					fl.append(int(n))
			if isinstance(fl[-1],float):
				i = fl[0]
				if i not in ovdict:
					ovdict[i] = fl[1:]
				else:
					ovdict[i] += fl[1:]
	print("%sOverlap Matrix " % YELLOW, end='')
	print("%sx-elements: %s%s, %sy-elements: %s%s%s" \
			% (YELLOW,GREEN,len(ovdict),YELLOW,GREEN,len(ovdict[0]),RS) )
	ovmatrix = []
	for i in range(0, len(ovdict)):
		ovmatrix.append(ovdict[i])
	return np.array(ovmatrix,np.float)

def fock(fh):
	fh.seek(0)
	key = 'Fock matrix for operator'
	fockdict = {}
	norb = [0]
	infock = False
	scfidx = 0
	print("Finding last Fock matrix...",end=' ')
	sys.stdout.flush()
	for l in fh:
		l = l.strip()
		if key in l:
			scfidx += 1
	print(scfidx)
	fh.seek(0)
	iscf = 0 
	print("Parsing Fock matrix...",end='')
	sys.stdout.flush()
	for l in fh:
		l = l.strip()
		if key in l:
			iscf += 1
			if not iscf % 5:
				sys.stdout.write('.')
				sys.stdout.flush()

		if key in l and not infock:
			infock = True
			continue
		elif key in l and infock:
			fockdict = {}
			norb = [0]
			continue
		elif '*' in l and infock:
			infock = False
			continue
		if iscf < scfidx:
			continue
		if infock:
			ditch = False
			lsf = filter(None, re.split('\s',l))
			fl = []
			for n in lsf:
				if '.' in n:
					try:
						n = float(n)
						if n == 0:
							n = abs(n)
						fl.append(n)
					except ValueError as msg:
						print(msg)
						ditch=True
						continue
				else:
					n = int(n)
					fl.append(n)
			if ditch:
				continue
			if isinstance(fl[-1],float) and isinstance(fl[0],int):
				norb.append(fl[0])
				if 0 < norb[-1] <= norb[-2]:
					#print("Out-of-order orbital: %s <= %s" % (norb[-1],norb[-2]))
					continue
				i = fl[0]
				if i not in fockdict:
					fockdict[i] = fl[1:]
				else:
					fockdict[i] += fl[1:]
	print('.')
	fockmatrix = []
	for i in sorted(fockdict.keys()):
		if len(fockdict[i]) != len(fockdict):
			print("Matrix alignment error: {%s} " % i, end='')
			sys.exit()
		else:
			fockmatrix.append(fockdict[i])
	fm = np.array(fockmatrix, np.float)
	if 0 in fm.shape:
		print("Empty Fock Matrix! ",end='')
		print(fm.shape)
		sys.exit()
	print("%sFock matrix " % YELLOW,end='')
	print("%sx-elements: %s%s, %sy-elements: %s%s%s" \
			% (YELLOW,GREEN,fm.shape[0],YELLOW,GREEN,fm.shape[1],RS) )
	return fm

def norbs(fh):
	fh.seek(0)
	orbdict = {}
	inorb = False
	orbidx = []
	lk = []
	rp = re.compile('^\d+\D+$')
	rps = re.compile('\s')
	lidx = 0
	print("Parsing molecular orbitals...",end='')
	for l in fh:
		lidx += 1
		if not lidx%1000000:
			sys.stdout.write(".")
			sys.stdout.flush()
		if not inorb:
			lk.append(l.strip())
			if len(lk) > 3:
				lk.pop(0)
			else:
				continue
		if lk[1] == 'MOLECULAR ORBITALS' and not inorb: 
			inorb = True
		elif l[0] == '*' and inorb:
			inorb = False
			break
		if inorb:
			lsf = []
			for ln in filter(None, re.split(rps,l)):
				lsf.append(ln)
			if not len(lsf):
				continue
			elif re.match(rp,lsf[0]) == None:
				continue
			if lsf[0] in orbdict:
				if lsf[1] in orbdict[lsf[0]]:
					continue
				else:
					orbdict[lsf[0]].append(lsf[1])
			else:
				orbdict[lsf[0]] = [lsf[1]]
				orbidx.append(lsf[0])
	sys.stdout.write("\n")
	torbs = 0
	for a in orbdict:
		torbs+=len(orbdict[a])
	if torbs == 0:
		print("%sNo orbitals found, caclulation probably did not converge!%s" % (RED,RS))
		sys.exit()
	else:
		print("%sAtoms: %s%s %sOrbitals: %s%s%s" \
				% (YELLOW,TEAL,len(orbdict),YELLOW,GREEN,torbs,RS) )
	return orbdict, orbidx

if __name__ == "__main__":
	if len(sys.argv) > 1:
		fn = sys.argv[1]
	if not os.path.isfile(fn):
		print("I can't read %s" % fn)
		sys.exit()
	with open(fn, 'rt') as fh:
		fm = fock(fh)
		orbs = norbs(fh)
		ol = overlap(fh)
