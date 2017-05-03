#!/usr/bin/env python3

import sys,os,re,argparse,subprocess

hn = os.uname()[1]
OSNAME=os.uname()[0]
if OSNAME == 'Darwin':
    p = subprocess.run(['find', '/Applications', '-maxdepth', '3', '-type', 'd', '-name', 'VMD*app'],stdout=subprocess.PIPE)
    VMDBIN=p.stdout.strip().split(b'\n')[-1]

elif OSNAME == 'Linux':
    p = subprocess.run(['which','vmd'],stdout=subprocess.PIPE)
    VMDBIN=p.stdout.strip()
else:
    VMDBIN=None

ORCABIN=None
for b in ('orca', 'orca.sh'):
    p = subprocess.run(['which', b],stdout=subprocess.PIPE)
    if p.returncode == 0:
        ORCABIN=p.stdout.strip()

if VMDBIN:
    print("Found VMD: %s" % VMDBIN)
if ORCABIN:
    print("Found Orca: %s" % ORCABIN)

def GetOrbsOrca(fn):
    orbs = []
    with open(fn, 'r') as fh:
        indef = False
        inorb = False
        for l in fh:
            if 'INPUT FILE' in l:
                indef = True
            elif '****END OF INPUT****' in l:
                indef = False
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
            #   inorb = False
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
        if fn[:-4] == fs[:-4] and 'gbw' in fs.lower():
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

def writeVMD(fn):
    with open(fn,'wt') as fh:
        mols = -1
        for o in opts.orbs:
            fh.write('mol new %s/%s.cube\n' % (os.getcwd(),o))
            mols += 1
        fh.write('rotate y by 90\n')
        fh.write('axes location off\n')
        fh.write('display projection orthographic\n')
        fh.write('mol addrep 0\n')
        fh.write('mol modstyle 0 0 Licorice\n')
        fh.write('mol modselect 0 0 all not name au\n')
        fh.write('mol modstyle 1 0 VDW\n')
        fh.write('mol modselect 1 0 all name au\n')
        fh.write('mol modcolor 1 0 Element\n')
        for m in range(0,mols+1): 
            if m == 0: i = 2
            else: i = 0
            fh.write('mol addrep %s\n' % m) 
            fh.write('mol modstyle %s %s Isosurface 0.003 0 0 0\n' % (i,m))
            fh.write('mol modcolor %s %s ColorID 0\n' % (i,m))
            fh.write('mol modmaterial %s %s Translucent\n' % (i,m))
            fh.write('mol addrep %s\n' % m) 
            fh.write('mol modstyle %s %s Isosurface -0.003 0 0 0\n' % (i+1,m))
            fh.write('mol modcolor %s %s ColorID 1\n' % (i+1,m))
            fh.write('mol modmaterial %s %s Translucent\n' % (i+1,m))
        if mols > 0:
            for m in range(1,mols+1):
                fh.write('mol delrep 2 %s\n' % m)
                fh.write('mol showrep %s %s off\n' % (m,0))
                fh.write('mol showrep %s %s off\n' % (m,1))
        #fh.write('render Wavefront /Volumes/Data/rchiechi/Calculations/orca/AQ/aqhomo.obj false\n')
    print('Wrote %s' % fn)

# # # # # # # # # # # # 

parser = argparse.ArgumentParser(description='Find orbitals in Orca and NWChem outputs'\
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infiles', type=str, nargs='*', default=[], 
    help='Output file to parse.')
parser.add_argument('-o', '--orbs', type=str, nargs='*', default=['HOMO','LUMO'], 
    help='Orbitals to render (e.g., HOMO-1, LUMO+1.')
parser.add_argument('-r','--render', action='store_true',  default=False, 
    help='If the appropriate programs are found, render the orbitals automatically.')
parser.add_argument('-g','--gbw', type=str,  default='', 
    help='Manually specify a GBW file instead of guessing from output file.')

opts=parser.parse_args()

if not opts.infiles:
    print("No input file.")
    sys.exit()

for fn in opts.infiles:
    ORBS={}
        
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
            if opts.gbw:
                gbw = opts.gbw
            else:
                gbw = FindGBW(fn)
            if gbw:
                print('gbw file: %s' % gbw)
            else:
                print('No GBW file found, try specifying it manually')
                continue
        except FileNotFoundError:
            print("%s does not exist." % fn)
            continue
        for i in range(0,len(orbs)):
            if orbs[i][1] == 0:
                for o in opts.orbs:
                    if o.upper() == 'HOMO':
                        ORBS[o] = (orbs[i-1][0],orbs[i-1][3])
                    if 'HOMO-' in o.upper():
                        offset = int(o.split('-')[-1])+1
                        ORBS[o] = (orbs[i-offset][0],orbs[i-offset][3])
                    if o.upper() == 'LUMO':
                        ORBS[o] = (orbs[i][0],orbs[i][3])
                    if 'LUMO+' in o.upper():
                        offset = int(o.split('+')[-1])
                        ORBS[o] = (orbs[i+offset][0],orbs[i+offset][3])
                break
        
        for o in ORBS:
            print('%s: (%0.4f eV)' % (o,ORBS[o][1]))
        print('# # # # # # # # # # # # # # # # # # # # # # # #')
        fn = '%s_plot.inp' % BN
        with open(fn, 'wt') as fh:
            fh.write('! DFT B3LYP/G LANL2DZ MOREAD NOITER\n')
            fh.write('# orca 3 ! Quick-DFT ECP{LANL2,LANLDZ} MOREAD NOITER\n')
            fh.write('* xyzfile 0 1 %s.xyz\n' % gbw[:-4])
            fh.write('%%base "%s-plot"\n' % BN)
            fh.write('%%MoInp "%s"\n' % gbw)
            fh.write('%plots\n')
            fh.write('dim1  128   # resolution in x-direction\n')
            fh.write('dim2  128   # resolution in y-direction\n')
            fh.write('dim3  128   # resolution in z-direction\n')
            fh.write('Format Gaussian_Cube\n')
            for o in ORBS:
                fh.write('MO("%s.cube",%s,0);  # orbital to plot\n' % (o,ORBS[o][0]))
            fh.write('end\n')
        print("Write %s" % fn)
        tclfn = fn[:-4]+'_vmd.tcl'
        writeVMD(tclfn)
        orcasuccess=False
        if ORCABIN and opts.render:
            print('Starting orca...')
            p = subprocess.run([ORCABIN,fn],stdout=subprocess.PIPE)
            if b'****ORCA TERMINATED NORMALLY****' in p.stdout:
                print('Successfully generated cube files with Orca.')
                orcasuccess=True
            else:
                print('Something may have gone wrong with Orca, check the output.')
                print('\n'.join(p.stdout.strip().split(b'\n')))
        if VMDBIN and orcasuccess:
            if OSNAME == 'Darwin':
                print("I am going to launch VMD now, but you probably have to copy/paste this command into the terminal:")
                print("source %s" % os.path.join(os.getcwd(),tclfn))
                subprocess.run(['open',VMDBIN, '--args', '-e', tclfn])
            else:
                subprocess.run([VMDBIN, '-e', tclfn])
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
