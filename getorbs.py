#!/usr/bin/env python


import sys,os,re


hn = os.uname()[1]

if 'pg' in hn or 'peregrine' in hn:
	ORCABIN=''
else:
	ORCABIN=os.path.join(os.environ['HOME'],'orca','bin')
#print("Orca bin dir is %s" % ORCABIN)

def GetOrbsOrca(fn):
	orbs = []
	with open(fn, 'r') as fh:
		inorb = False
		for l in fh.readlines():
			if 'ORBITAL ENERGIES' in l:
				inorb = True
			elif '------------------' in l:
				inorb = False
			if inorb:
				try:
					#   NO   OCC          E(Eh)            E(eV)
					lo = re.split('\s+', l.strip())
					orbs.append( [int(lo[0])]+list( map(float, lo[1:]) ) )
				except ValueError as msg:
					#print(msg)
					#print("Error finding orbital in %s" % l)
					continue

	return orbs

def GetOrbsNwchem(fn):
	orbs = []
	with open(fn, 'r') as fh:
		inorb = False
		for l in fh.readlines():
			if 'Vector' in l:
				inorb = True
			#elif l == '':
			#	inorb = False
			else:
				inorb = False
			if inorb:
				try:
					#Vector 1363  Occ=0.000000D+00  E= 2.383287D+01  Symmetry=bu
					lo = re.split('\s+', l.strip())
					#orbs.append(list(map(int, lo)))
					orbs.append(lo)
				except ValueError:
					#print("Error finding orbital in %s" % l)
					continue

	return orbs

def FindGBW(fn):
	d,f = os.path.split(fn)
	if not d:
		d = './'
	for fs in os.listdir(d):
		if f[:-4] in fs and 'gbw' in fs.lower():
			return fs

def writedplot(bn, homo,heng,lumo,leng):

	def gen(title, orb, eng):
		dplot = ['dplot' ,'  title "%s (%0.4f eV)"' % (title,eng*27.207), '  gaussian',
			 '  spin total',
		         '  orbitals view; 1; %s' % orb, 
			 '  output "%s_%s_%s.cube"' % (bn, title, orb),
			 '  limitXYZ', '  -20.0 20.0 200', 
			 '  -20.0 20.0 200', '  -20.0 20.0 200',
			 'end\n']
		return "\n".join(dplot)

	dplot_homo = gen('HOMO', homo, heng)
	dplot_lumo = gen('LUMO', lumo, leng)

	return dplot_homo, dplot_lumo

if not len(sys.argv) > 1:
	print("No input file.")
	sys.exit()

#if 'orca' in sys.argv[1].lower():
#	PROG='orca'
#elif 'nwchem' in sys.argv[1].lower():
#	PROG='nwchem'
#else:
#	print("What program?")
#	sys.exit()

for fn in sys.argv[1:]:
	with open(fn, 'rt') as fh:
		h = fh.read(2048)
		if 'nwchem' in h.lower():
			PROG = 'nwchem'
			
		elif 'O   R   C   A' in h:
			PROG = 'orca'
		else:
			print("I don't know what kind of file this is.")
			continue
	print("# # # # # # # # %s (%s) # # # # # # # #" % (fn,PROG))
	BN = os.path.basename(fn)[:-4]
	HOMO,LUMO=0,0
	heng,leng=0,0
	if PROG=='orca':
		try:
			orbs = GetOrbsOrca(fn)
			gbw = FindGBW(fn)
		except FileNotFoundError:
			print("%s does not exist." % fn)
			continue
		for i in range(0,len(orbs)):
			if orbs[i][1] == 0:
				HOMO = orbs[i-1][0]
				heng = orbs[i-1][3]
				LUMO = orbs[i][0]
				leng = orbs[i][3]
				break
		#print('HOMO: %s, LUMO: %s' % (HOMO,LUMO))
		print('HOMO: %s (%0.4f eV), LUMO: %s (%0.4f eV)' % (HOMO,heng,LUMO,leng) )
		print('# # # # # # # # # # # # # # # # # # # # # # # #')
		if ORCABIN:
			os.system('%s/orca_plot %s -i' % (ORCABIN,gbw))
		else:
			os.system('module add ORCA; orca_plot %s -i' % gbw)
	elif PROG == 'nwchem':
		try:
			orbs = GetOrbsNwchem(fn)
		except FileNotFoundError:
			print("%s does not exist." % fn)
			continue
		for i in range(0, len(orbs)):
			#Vector 1363  Occ=0.000000D+00  E= 2.383287D+01  Symmetry=bu
			orbnum = int(orbs[i][1])
			orbocc = float(orbs[i][2].split('=')[1].replace('D','E'))
			if orbocc == 0:
				HOMO = int(orbs[i-1][1])
				LUMO = orbnum
				heng = float(orbs[i-1][3].split('=')[1].replace('D','E'))
				leng = float(orbs[i][3].split('=')[1].replace('D','E'))
				break
		print("# # # %s_orbs.nw # # #" % BN)
		dplot_homo, dplot_lumo = writedplot(BN, HOMO,heng,LUMO,leng)
		with open(BN+'_orbs.nw', 'wt') as fh:
			fh.write('restart %s\n' % BN)
			fh.write('title %s_orbs\n' % BN)
			fh.write('memory 1 GB\n\n\n')
			fh.write(dplot_homo)
			fh.write('task dplot\n\n')
			fh.write(dplot_lumo)
			fh.write('task dplot\n\n')

		print('HOMO: %s (%0.4f eV), LUMO: %s (%0.4f eV)' % (HOMO,heng*27.2107,LUMO,leng*27.2107))
