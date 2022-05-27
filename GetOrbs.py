#!/usr/bin/env python3
'''
Version: 1.0
Copyright (C) 2022 Ryan Chiechi <ryan.chiechi@ncsu.edu>
Description:
        This program parses the outputs of quantum chemistry programs
        and renders isoplots as cube files using VMD. It is mostly useful
        for processing ORCA outputs.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

ORCA eplot: (c) 2013 Marius Retegan
License: BSD-2-Clause

'''
import sys
import os
import re
import json
import argparse
import subprocess
import configparser
from collections import OrderedDict
# try:
#    import pip
# except ImportError:
#    print('You don\'t have pip installed. You will need pip to istall other dependencies.')
#    sys.exit(1)

# prog = os.path.basename(sys.argv[0]).replace('.py','')

# installed = [package.project_name for package in pip.get_installed_distributions()]
# required = ['numpy','colorama','psutil']
# for pkg in required:
#    if pkg not in installed:
#        print('You need to install %s to use %s.' % (pkg,prog))
#        print('e.g., sudo -H pip3 install --upgrade %s' % pkg)
#        sys.exit(1)


reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
prog = os.path.basename(sys.argv[0]).replace('.py','')

required = ['numpy','colorama']
for pkg in required:
    if pkg not in installed_packages:
        print('You need to install %s to use %s.' % (pkg,prog))
        print('e.g., sudo -H pip3 install --upgrade %s' % pkg)
        sys.exit(1)

try:
    # from psutil import cpu_count
    import numpy as np
    from colorama import init,Fore,Back,Style
except ModuleNotFoundError as msg:
    print('You need to install additional pacakges, e.g., sudo -H pip3 install --upgrade <package>')
    print(msg)
    sys.exit(1)

# Setup colors
init(autoreset=True)


ELEMENTS = [None,
            "H", "He",
            "Li", "Be",
            "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg",
            "Al", "Si", "P", "S", "Cl", "Ar",
            "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe",
            "Cs", "Ba",
            "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn",
            "Fr", "Ra",
            "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
            "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub"]

# # # # # # # # # # # # # # # # FUNCTIONS  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def FindBins():
    # Find paths to binaries
    if OSNAME == 'Darwin':
        p = subprocess.run(['find', '/Applications', '-maxdepth', '3', '-type', 'd', '-name', 'VMD*app'],stdout=subprocess.PIPE)
        vmdbin = os.path.join(p.stdout.strip().split(b'\n')[-1],b'Contents/MacOS/startup.command')

    elif OSNAME == 'Linux':
        p = subprocess.run(['which','vmd'],stdout=subprocess.PIPE)
        vmdbin = p.stdout.strip()
    else:
        vmdbin = None

    orcabin = ''
    for b in ('orca', 'orca.sh'):
        p = subprocess.run(['which', b],stdout=subprocess.PIPE)
        if p.returncode == 0:
            orcabin = p.stdout.strip()
    vpotbin = ''
    p = subprocess.run(['which', 'orca_vpot'],stdout=subprocess.PIPE)
    if p.returncode == 0:
        vpotbin = p.stdout.strip()

    if os.path.exists(vmdbin):
        print(Fore.YELLOW+"Found VMD: %s" % vmdbin)
    else:
        print(Fore.RED+"VMD was not found, specify the path if you want to render isoplots automatically.")
    if os.path.exists(orcabin):
        print(Fore.YELLOW+"Found Orca: %s" % orcabin)
    else:
        print(Fore.RED+"Orca was not found, specify the path if you want to generate cube files automatically.")
    if os.path.exists(vpotbin):
        print(Fore.YELLOW+"Found orca_vpot: %s" % vpotbin)
    else:
        print(Fore.RED+"orca_vpot was not found, specify the path if you want to compute eplots.")
    return orcabin,vpotbin,vmdbin

def checkorbs(_l, indef, inorbs, inalphaorb, inbetaorb):
    if 'INPUT FILE' in _l:
        indef = True
    elif '****END OF INPUT****' in _l:
        indef = False
    if 'ORBITAL ENERGIES' in _l:
        inorbs = True
    if 'SPIN UP ORBITALS' in _l:
        inalphaorb = True
        inbetaorb = False
    if 'SPIN DOWN ORBITALS' in _l:
        inalphaorb = False
        inbetaorb = True
    elif ('------------------' in _l) or ('********' in _l):
        inbetaorb, inalphaorb, inorbs = False, False, False
    return indef, inorbs, inalphaorb, inbetaorb

def GetOrbsOrca(fn,opts):
    orbs = {'nospin':[], 'alpha':[], 'beta':[]}
    dft = ''
    with open(fn, 'r') as fh:
        indef, inorbs, inalphaorb, inbetaorb = False, True, False, False
        for _l in fh:
            indef, inorbs, inalphaorb, inbetaorb = checkorbs(_l, indef, inorbs, inalphaorb, inbetaorb)
            if indef:
                if _l.strip()[0] == '|':
                    for c in _l:
                        if c == '#':
                            break
                        if c == '!':
                            dft = '! '+_l.split('!')[-1].strip()
                            # if 'PAL' not in dft and opts.pal:
                            #     dft += f" PAL{opts.pal}"
                            if 'MOREAD' not in dft:
                                dft += " MOREAD"
                            break

            if inalphaorb or inbetaorb or inorbs:
                if inalphaorb:
                    key = 'alpha'
                elif inbetaorb:
                    key = 'beta'
                else:
                    key = 'nospin'
                try:
                    #   NO   OCC          E(Eh)            E(eV)
                    lo = re.split('\s+', _l.strip())  # noqa
                    orbs[key].append([int(lo[0])]+list(map(float, lo[1:])))
                except ValueError:
                    continue

    return orbs,dft


def FindGBW(fn):
    d,f = os.path.split(fn)
    if not d:
        d = './'
    for fs in os.listdir(d):
        if fn[:-4] == fs[:-4] and 'gbw' in fs.lower():
            return fs


def read_xyz(xyz):
    atoms = []
    x = []
    y = []
    z = []
    with open(xyz, "r") as fh:
        next(fh)
        next(fh)
        for line in fh:
            data = line.split()
            atoms.append(data[0])
            x.append(float(data[1]))
            y.append(float(data[2]))
            z.append(float(data[3]))
    return atoms, np.array(x), np.array(y), np.array(z)

def OrcaEplot(BN, gbw, rccconfig, opts):

    vpotbin = rcconfig['GENERAL']['ORCAvpot']

    npoints = opts.eplotres

    ang_to_au = 1.0 / 0.5291772083

    atoms, x, y, z = read_xyz("%s.xyz" % BN)
#    natoms = len(atoms)

    extent = 7.0
    xmin = x.min() * ang_to_au - extent
    xmax = x.max() * ang_to_au + extent
    ymin = y.min() * ang_to_au - extent
    ymax = y.max() * ang_to_au + extent
    zmin = z.min() * ang_to_au - extent
    zmax = z.max() * ang_to_au + extent

    with open(f"{BN}_eplot.xyz", "w") as mep_xyz:
        mep_xyz.write("{0:d}\n".format(npoints**3))
        for ix in np.linspace(xmin, xmax, npoints, True):
            for iy in np.linspace(ymin, ymax, npoints, True):
                for iz in np.linspace(zmin, zmax, npoints, True):
                    mep_xyz.write("{0:12.6f} {1:12.6f} {2:12.6f}\n".format(ix, iy, iz))
    with open(f"{BN}_eplot.inp", "w") as mep_inp:
        mep_inp.write(f'{str(opts.pal)}  # Number of parallel processes\n')
        mep_inp.write(f'{BN}-plot.gbw  # GBW File\n')
        mep_inp.write(f'{BN}-plot.scfp  # Density\n')
        mep_inp.write(f'{BN}_eplot.xyz  # Coordinates\n')
        mep_inp.write(f'{BN}_eplot.out  # Output File\n')
        # mep_inp.write(f'{BN}-plot   # BaseName of DensityContainer\n')

    # try:
    #     subprocess.check_call([vpotbin, f"{BN}_eplot.inp"])
    print(Fore.CYAN+'Starting vpot...')
    p = subprocess.run([vpotbin, f"{BN}_eplot.inp"], stdout=subprocess.PIPE, env=ENV)
    if p.returncode == 0:
        print(f'{Fore.GREEN} Done!')
    else:
        print(f'{Fore.RED}{Style.BRIGHT}vpot job may have failed.')
        _lines = str(p.stdout, encoding='utf8')
        print(f'{Fore.RED}{_lines}')
    # except subprocess.CalledProcessError:
        # print(Fore.RED+Style.BRIGHT+"orca_vpot returned an error, cannot generate eplot cube.")
        # return
    if not os.path.exists(f'{BN}_eplot.out'):
        print(f'{Fore.RED}{Style.BRIGHT}Failed to generate {BN}_eplot.out')
        return
    with open(f"{BN}_eplot.out", "r") as fh:
        _v = []
        next(fh)
        for line in fh:
            _data = line.split()
            _v.append(float(_data[3]))
        vpot = np.array(_v)

    with open("%s_eplot.cube" % BN, "w") as cube:
        cube.write("Generated with ORCA\n")
        cube.write("Electrostatic potential for " + BN + "\n")
        cube.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(
            len(atoms), xmin, ymin, zmin))
        cube.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(
            npoints, (xmax - xmin) / float(npoints - 1), 0.0, 0.0))
        cube.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(
            npoints, 0.0, (ymax - ymin) / float(npoints - 1), 0.0))
        cube.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(
            npoints, 0.0, 0.0, (zmax - zmin) / float(npoints - 1)))
        for i, atom in enumerate(atoms):
            index = ELEMENTS.index(atom)
            cube.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n".format(
                index, 0.0, x[i] * ang_to_au, y[i] * ang_to_au, z[i] * ang_to_au))

        m = 0
        n = 0
        vpot = np.reshape(vpot, (npoints, npoints, npoints))
        for ix in range(npoints):
            for iy in range(npoints):
                for iz in range(npoints):
                    cube.write("{0:14.5e}".format(vpot[ix][iy][iz]))
                    m += 1
                    n += 1
                    if (n > 5):
                        cube.write("\n")
                        n = 0
                if n != 0:
                    cube.write("\n")
                    n = 0
    print(f"{Fore.GREEN}Wrote eplot to %s_eplot.cube" % BN)

def writePymol(BN):
    with open('makeorbs.pml', 'wt') as fh:
        fh.write(f'load "{BN}.xyz",XYZ\n')
        fh.write('as sticks, XYZ\n')
        fh.write('util.cbaw\n')
        fh.write(f'load "{BN}_HOMO.cube",HOMO\n')
        fh.write(f'load "{BN}_LUMO.cube",LUMO\n')
        fh.write(f'load "{BN}_eplot.cube",eplot\n')
        fh.write('''
cmd.volume_ramp_new('eldens', [\\
0.02, 0.00, 0.00, 1.00, 0.00, \\
0.03, 0.00, 1.00, 1.00, 0.20, \\
0.06, 0.00, 0.00, 1.00, 0.00, \\
])
cmd.volume_ramp_new('eplot', [\\
-0.04, 0.00, 0.00, 1.00, 0.12, \\
-0.01, 0.00, 1.00, 1.00, 0.00, \\
0.14, 0.00, 1.00, 0.00, 0.02, \\
0.53, 1.00, 1.00, 0.00, 0.06, \\
1.00, 1.00, 0.50, 0.00, 0.01, \\
1.82, 1.00, 0.00, 0.00, 0.09, \\
])
cmd.volume_ramp_new('homo', [\\
-0.005, 1.00, 0.00, 0.00, 0.30, \\
0.00, 0.96, 0.12, 0.80, 0.00, \\
0.00, 0.00, 0.98, 0.93, 0.00, \\
0.005, 0.00, 0.00, 1.00, 0.30, \\
])
cmd.volume_ramp_new('lumo', [\\
-0.005, 1.00, 1.00, 0.00, 0.30, \\
0.00, 0.00, 1.00, 0.00, 0.00, \\
0.00, 0.00, 0.00, 1.00, 0.00, \\
0.005, 0.00, 1.00, 1.00, 0.30, \\
    ])
as sticks,XYZ
util.cbaw
volume HOMO_volume, HOMO, homo
disable HOMO_volume
volume LUMO_volume, LUMO, lumo
disable LUMO_volume
volume eplot_volume, eplot, eplot
disable eplot_volume
isosurface HOMO_iso, HOMO, 0.005
disable HOMO_iso
isosurface LUMO_iso, LUMO, 0.005
disable LUMO_iso
isosurface eplot_iso, eplot, 0.01
disable eplot_iso
set transparency, 0.2
        ''')

def writeVMD(fn,opts,BN):
    with open(fn,'wt') as fh:
        mols = -1
        for o in ORBS:
            # fh.write('mol new "%s/%s.cube"\n' % (os.getcwd(),ORBS[o][2]))
            fh.write('mol new "%s.cube"\n' % ORBS[o][2])
            mols += 1
            fh.write('mol rename %s %s\n' % (mols,o))
        if opts.eplot:
            # fh.write('mol new "%s/%s_eplot.cube"\n' % (os.getcwd(),BN))
            fh.write('mol new "%s_eplot.cube"\n' % BN)
            fh.write('mol rename %s eplot\n' % (mols+1))
        if opts.spindens:
            # fh.write('mol new "%s/%s_spindens.cube"\n' % (os.getcwd(),BN))
            fh.write('mol new "%s_spindens.cube"\n' % BN)
            fh.write('mol rename %s spindens\n' % (mols+2))
        fh.write('rotate y by 90\n')
        fh.write('axes location off\n')
        fh.write('display projection orthographic\n')
        fh.write('mol addrep 0\n')
        fh.write('mol modstyle 0 0 %s 0.3 37 37\n' % opts.molmethod)
        fh.write('mol modselect 0 0 all not name %s\n' % opts.electrode)
        fh.write('mol modstyle 1 0 %s 1 37\n' % opts.electrodemethod)
        fh.write('mol modselect 1 0 all name %s\n' % opts.electrode)
        fh.write('mol modcolor 1 0 Element\n')

        _max = 1
        if opts.eplot:
            _max = 2
        if opts.spindens:
            _max = 3

        for m in range(0,mols+_max):
            if m == 0:
                i = 2
            else:
                i = 0
            fh.write('mol addrep %s\n' % m)
            fh.write('mol modstyle %s %s Isosurface %s 0 0 0\n' % (i,m,opts.isovalue))
            fh.write('mol modcolor %s %s ColorID %s\n' % (i,m,VMDCOLORS[opts.colors[0]]))
            fh.write('mol modmaterial %s %s %s\n' % (i,m,opts.material))
            fh.write('mol addrep %s\n' % m)
            fh.write('mol modstyle %s %s Isosurface -%s 0 0 0\n' % (i+1,m,opts.isovalue))
            fh.write('mol modcolor %s %s ColorID %s\n' % (i+1,m,VMDCOLORS[opts.colors[1]]))
            fh.write('mol modmaterial %s %s %s\n' % (i+1,m,opts.material))
        if mols > 0:
            for m in range(1,mols+_max):
                fh.write('mol delrep 2 %s\n' % m)
                fh.write('mol showrep %s %s off\n' % (m,0))
                fh.write('mol showrep %s %s off\n' % (m,1))
        fh.write('menu graphics on\n')
    print(Fore.GREEN+Style.BRIGHT+'Wrote %s' % fn)


def writeSimpleVMD(fn,xyz):
    with open(fn,'wt') as fh:
        fh.write('mol new "%s/%s"\n' % (os.getcwd(),xyz))
        fh.write('rotate x by -90\n')
        fh.write('axes location off\n')
        fh.write('display projection orthographic\n')
        fh.write('mol addrep 0\n')
        fh.write('mol modstyle 0 0 %s 0.3 37 37\n' % opts.molmethod)
        fh.write('mol modselect 0 0 all not name %s\n' % opts.electrode)
        fh.write('mol modstyle 1 0 %s 1 37\n' % opts.electrodemethod)
        fh.write('mol modselect 1 0 all name %s\n' % opts.electrode)
        fh.write('mol modcolor 1 0 Element\n')
        fh.write('menu graphics on\n')
    print(Fore.GREEN+Style.BRIGHT+'Wrote %s' % fn)


def writeCubeVMD(fn,cube):
    with open(fn,'wt') as fh:
        fh.write('mol new "%s/%s"\n' % (os.getcwd(),cube))
        fh.write('rotate x by -90\n')
        fh.write('axes location off\n')
        fh.write('display projection orthographic\n')
        fh.write('mol addrep 0\n')
        fh.write('mol modstyle 0 0 %s 0.3 37 37\n' % opts.molmethod)
        fh.write('mol modselect 0 0 all not name %s\n' % opts.electrode)
        fh.write('mol modstyle 1 0 %s 1 37\n' % opts.electrodemethod)
        fh.write('mol modselect 1 0 all name %s\n' % opts.electrode)
        fh.write('mol modcolor 1 0 Element\n')
        m = 0
        i = 2
        fh.write('mol addrep %s\n' % m)
        fh.write('mol modstyle %s %s Isosurface %s 0 0 0\n' % (i,m,opts.isovalue))
        fh.write('mol modcolor %s %s ColorID %s\n' % (i,m,VMDCOLORS[opts.colors[0]]))
        fh.write('mol modmaterial %s %s %s\n' % (i,m,opts.material))
        fh.write('mol addrep %s\n' % m)
        fh.write('mol modstyle %s %s Isosurface -%s 0 0 0\n' % (i+1,m,opts.isovalue))
        fh.write('mol modcolor %s %s ColorID %s\n' % (i+1,m,VMDCOLORS[opts.colors[1]]))
        fh.write('mol modmaterial %s %s %s\n' % (i+1,m,opts.material))
        fh.write('menu graphics on\n')
        fh.write('color Display Background white\n')
        fh.write('color Display FPS black\n')
        fh.write('color Axes Labels black\n')

    print(Fore.GREEN+Style.BRIGHT+'Wrote %s' % fn)


def getprog(fn):
    PROG = None
    ext = fn.split('.')[-1]
    with open(fn, 'rt') as fh:
        h = fh.read(2048)
        if 'O   R   C   A' in h:
            PROG = 'orca'
            print(Fore.YELLOW+"Parsing ORCA output file")
        elif fn[-3:].lower() == 'xyz':
            print(Fore.YELLOW+"Parsing XYZ file")
            try:
                int(h[0])
                PROG = 'xyz'
            except ValueError:
                pass
        elif ext.lower() in ('cub', 'cgube', 'cube'):
            print(Fore.YELLOW+"Parsing cube file")
            try:
                int(h[0])
            except ValueError:
                PROG = 'cube'
    return PROG

def printorcaorbs(key):
    __orbs = key
    for i in range(0,len(__orbs)):
        if __orbs[i][1] == 0:
            for o in opts.orbs:
                if o.upper() == 'LUMO':
                    ORBS[o] = (__orbs[i][0],__orbs[i][3],BN+'_'+o)
                if 'LUMO+' in o.upper():
                    offset = int(o.split('+')[-1])
                    ORBS[o] = (__orbs[i+offset][0],__orbs[i+offset][3],BN+'_'+o)
                if o.upper() == 'HOMO':
                    ORBS[o] = (__orbs[i-1][0],__orbs[i-1][3],BN+'_'+o)
                if 'HOMO-' in o.upper():
                    offset = int(o.split('-')[-1])+1
                    ORBS[o] = (__orbs[i-offset][0],__orbs[i-offset][3],BN+'_'+o)
            break
    if not ORBS and (not opts.eplot and not opts.spindens):
        print(Fore.RED+"No orbitals selected!")
        return
    for o in ORBS:
        print(Style.BRIGHT+'%s: (%0.4f eV)' % (o,ORBS[o][1]))

def doorcaproc(opts, gbw, fn, tclfn, runorca):
    orcasuccess = False
    print(Fore.BLUE+Back.WHITE+'# # # # # # # # Render Cube  # # # # # # # # # # # #')
    if ((opts.render and opts.ORCApath) or opts.eplot or opts.spindens) and runorca:
        print(Fore.CYAN+'Starting orca...')
        p = subprocess.run([opts.ORCApath,fn],stdout=subprocess.PIPE, env=ENV)
        if b'****ORCA TERMINATED NORMALLY****' in p.stdout:
            print(Fore.GREEN+'Successfully generated cube files with Orca.')
            orcasuccess = True
        else:
            print(Back.RED+'Something may have gone wrong with Orca, check the output:')
            _lines = str(p.stdout, encoding='utf8').split('\n')
            if len(_lines) > 15:
                _err = "\n".join(_lines[-15:])
            else:
                _err = "\n".join(_lines)
            print(f'{Fore.RED}{_err}')
    elif opts.ORCApath and opts.render:
        orcasuccess = True
        print(Fore.BLUE+"Skipping Orca run because cube files already exist.")
    print(Fore.BLUE+Back.WHITE+'# # # # # # # # # # # # # # # # # # # # # # # #')

    if orcasuccess and opts.eplot:
        if os.path.exists("%s_eplot.cube" % BN):
            print(Fore.BLUE+"Skipping vpot run because cube files already exist.")
        else:
            OrcaEplot(BN, gbw, rcconfig, opts)

    if opts.render and opts.VMDpath and orcasuccess:
        subprocess.run([opts.VMDpath, '-e', tclfn], env=ENV)

def doorcaprog(fn, opts):
    try:
        orbs,dft = GetOrbsOrca(fn,opts)
        if opts.gbw != 'guess':
            gbw = opts.gbw
        else:
            gbw = FindGBW(fn)
        if gbw:
            print(Fore.YELLOW+'gbw file: %s' % gbw)
        else:
            print(Fore.RED+Style.BRIGHT+'No GBW file found, try specifying it manually')
            pass
    except FileNotFoundError:
        print(Fore.RED+"%s does not exist." % fn)
        pass
    if orbs['alpha']:
        print('--- Alpha Spin ---')
        printorcaorbs(orbs['alpha'])
        print('--- Beta Spin ---')
        printorcaorbs(orbs['beta'])
    else:
        printorcaorbs(orbs['nospin'])

    print(Back.BLUE+Fore.WHITE+'# # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    RUNORCA = False
    fn_plot = f'{BN}_plot.inp'
    xyz = f'{BN}.xyz'
    if not os.path.exists(xyz):
        if os.path.exists(f'{gbw[:-4]}.xyz'):
            print(f'{Fore.YELLOW}Using {gbw[:-4]}.xyz as the XYZ file.')
            with open(f'{gbw[:-4]}.xyz', 'r') as fh:
                with open(xyz, 'w') as xyz_fh:
                    xyz_fh.write(fh.read())
        else:
            print(f'{Fore.RED}{Style.BRIGHT}Did not find {xyz}')
            sys.exit()
    with open(fn_plot, 'wt') as fh:
        fh.write('%s NOITER KEEPDENS\n' % dft)
        fh.write('#! DFT B3LYP/G LANL2DZ MOREAD NOITER PAL8\n')
        fh.write('# orca 3 ! Quick-DFT ECP{LANL2,LANLDZ} MOREAD NOITER\n')
        if opts.pal > 1:
            fh.write(f'%pal nprocs {opts.pal}\nend\n')
        fh.write('* xyzfile %s %s %s\n' % (opts.charge,opts.spin,xyz))
        fh.write('%%base "%s-plot"\n' % BN)
        fh.write('%%MoInp "%s"\n' % gbw)
        fh.write('%plots\n')
        fh.write('dim1  128   # resolution in x-direction\n')
        fh.write('dim2  128   # resolution in y-direction\n')
        fh.write('dim3  128   # resolution in z-direction\n')
        fh.write('Format Gaussian_Cube\n')
        if opts.eplot:
            fh.write('ElDens("%s_eldens.cube"); # Electron density\n' % BN)
            if not os.path.exists("%s_eldens.cube" % BN):
                RUNORCA = True
            if not os.path.exists(f"{BN}_eplot.cube"):
                RUNORCA = True
        if opts.spindens:
            fh.write('SpinDens("%s_spindens.cube"); # Spin density\n' % BN)
            if not os.path.exists("%s_spindens.cube" % BN):
                RUNORCA = True
        for o in ORBS:
            fh.write('MO("%s.cube",%s,0);  # orbital to plot\n' % (ORBS[o][2],ORBS[o][0]))
            if not os.path.exists("%s.cube" % ORBS[o][2]):
                RUNORCA = True
        fh.write('end\n')
    print(Fore.GREEN+Style.BRIGHT+"Wrote %s" % fn_plot)
    tclfn = fn_plot[:-4]+'_vmd.tcl'
    writeVMD(tclfn, opts, BN)
    doorcaproc(opts, gbw, fn_plot, tclfn, RUNORCA)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


RCFILE = os.path.join(os.path.expanduser('~'),'.getorbsrc')
OSNAME = os.uname()[0]
VMDCOLORS = {'blue':0,'red':1,'gray':2,'orange':3,'yellow':4,
             'tan':5,'silver':6,'green':7,'white':8,'pink':9,
             'cyan':10,'purple':11,'lime':12,'mauve':13,
             'ochre':14,'iceblue':15,'black':16,'yellow2':17,
             'yellow3':18,'green2':19,'green3':20,'cyan2':21,
             'cyan3':22,'blue2':23,'blue3':24,'violet':25,
             'violet2':26,'magenta':27,'magenta2':28,
             'red2':29,'red3':30,'orange2':31,'orange3':32}
VMDMATERIALS = ('Opaque','Transparent','BrushedMetal','Diffuse',
                'Ghost','Glass1','Glass2','Glass3','Glossy',
                'HardPlastic','MetallicPastel','Steel',
                'Translucent','Edgy','EdgyShiny','EdgyGlass',
                'Goodsell','AOShiny','AOChalky','AOEdgy',
                'BlownGlass','GlassBubble','RTChrome')
VMDMETHODS = ('Lines','Bonds','DynamicBonds','HBonds',
              'Points','VDW','CPK','Licorice','Polyhedra',
              'Trace','Tube','Ribbons','NewRibbons',
              'Cartoon','NewCartoon','PaperChain',
              'Twister','QuickSurf','Surf','MSMS',
              'VolumeSlice','Isosurface','FieldLines',
              'Orbital','Beads','Dotted','Solvent')
# Parse config file
rcconfig = configparser.ConfigParser()
if not rcconfig.read(RCFILE):
    orcabin,vpotbin,vmdbin = FindBins()
    rcconfig['GENERAL'] = {'ORCApath':orcabin,
                           'ORCAvpot': vpotbin,
                           'VMDpath':vmdbin,
                           'ENV':'{}',
                           'pal':'1',
                           'render':'no',
                           'orbs':'HOMO, LUMO'}
    rcconfig['VMD'] = {'colors':'blue, red',
                       'material':'Translucent',
                       'molmethod':'Licorice',
                       'electrodemethod':'VDW',
                       'isovalue':0.005,
                       'electrode':'Au'}
    with open(RCFILE,'w') as fh:
        rcconfig.write(fh)
    print(Fore.YELLOW+Style.BRIGHT+'I wrote default values to %s. Edit that file to change them.' % RCFILE)
else:
    print(Fore.GREEN+'Read defaults from %s' % RCFILE)

# Convert binary strings to paths
for path in ('VMDpath','ORCAvpot','ORCApath'):
    if "b'" not in rcconfig['GENERAL'][path]:
        continue
    m = re.match('^b?[\'"](.*)[\'"]', rcconfig['GENERAL'][path])
    if not m:
        print(Fore.RED+Style.BRIGHT+"Error parsing %s as a path." % rcconfig['GENERAL'][path])
    else:
        rcconfig['GENERAL'][path] = m.groups()[0].strip()

# Parse command line arguments
desc = 'Find and render orbitals in Orca outputs and cube files using VMD.\
        Default values are stored in %s. Edit this file to change them' % RCFILE
parser = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infiles', type=str, nargs='*', default=[],
                    help='Output file to parse.')
parser.add_argument('-o', '--orbs', type=str, nargs='*', default=rcconfig['GENERAL']['orbs'].replace(' ','').split(','),
                    help='Orbitals to render (e.g., HOMO-1, LUMO+1).')
parser.add_argument('--eplot', action='store_true', default=False,
                    help='Generate an electrostatic potential isoplot.')
parser.add_argument('--spindens', action='store_true', default=False,
                    help='Generate a spin density plot.')
parser.add_argument('--eplotres', type=int, default=40,
                    help='Grid density of eplot (80 will take forever, but give a smooth plot).')
parser.add_argument('-r','--render', action='store_true', default=rcconfig['GENERAL'].getboolean('render'),
                    help='If the appropriate programs are found, render the orbitals automatically.')
parser.add_argument('--pal', type=int, default=rcconfig['GENERAL']['pal'],
                    help='Use parallel/mpi Orca with n CPUs.')
parser.add_argument('-g','--gbw', type=str, default='guess',
                    help='Manually specify a GBW file instead of guessing from output file.')
parser.add_argument('-O','--ORCApath', type=str, default=rcconfig['GENERAL']['ORCApath'],
                    help='Specify the location of the orca binary.')
parser.add_argument('-V','--VMDpath', type=str, default=rcconfig['GENERAL']['VMDpath'],
                    help='Specify the location of the vmd binary.')
parser.add_argument('--env', type=str, default=rcconfig['GENERAL']['ENV'],
                    help='Environmental variables to load when calling binaries in JSON format, e.g. \'{"LD_LIBRARY_PATH": "/opt/orca/"}\'.')
parser.add_argument('-c','--colors', nargs=2, default=rcconfig['VMD']['colors'].replace(' ','').split(','), choices=tuple(VMDCOLORS.keys()),
                    help='Colors for +/- orbital coefficients.')
parser.add_argument('-m','--material', type=str, default=rcconfig['VMD']['material'].strip(), choices=VMDMATERIALS,
                    help='Material to use for isosurfaces.')
parser.add_argument('-M','--molmethod', type=str, default=rcconfig['VMD']['molmethod'].strip(), choices=VMDMETHODS,
                    help='Method used to render the molecule.')
parser.add_argument('-E','--electrodemethod', type=str, default=rcconfig['VMD']['electrodemethod'].strip(), choices=VMDMETHODS,
                    help='Material to use for electrode.')
parser.add_argument('-i','--isovalue', type=float, default=rcconfig['VMD'].getfloat('isovalue'),
                    help='Cutoff for isoplots.')
parser.add_argument('-e','--electrode', type=str, default=rcconfig['VMD']['electrode'].strip(),
                    help='Type of electrode, if present.')
parser.add_argument('-q','--charge', type=int, default=0,
                    help='Net charge on molecule.')
parser.add_argument('-s','--spin', type=int, default=1,
                    help='Spin multiplicity of molecule.')

opts = parser.parse_args()
ENV = os.environ
_env = json.loads(opts.env)
print(f'{Fore.YELLOW}Loaded environment: {_env}')
ENV.update(_env)

# Check that options were parsed correctly
if not opts.infiles:
    print(Fore.RED+"No input file.")
    sys.exit()

if len(opts.colors) != 2:
    print(Fore.RED+"Too many colors: %s" % str(opts.colors))
    sys.exit()

for orb in opts.orbs:
    if (',' in orb) or (' ' in orb):
        print(Fore.RED+"Something is wrong with the orbital specfication: %s" % str(opts.orbs))
        print(Fore.YELLOW+"Orbitals are specified like this: --orbs HOMO LUMO LUMO+1")
        sys.exit()

for c in opts.colors:
    if c not in VMDCOLORS.keys():
        print(Fore.RED+"Invalid color selection: %s" % str(opts.colors))
        sys.exit()
if opts.material not in VMDMATERIALS:
    print(Fore.RED+"Invalid material: %s" % str(opts.material))
    sys.exit()
if opts.molmethod not in VMDMETHODS:
    print(Fore.RED+"Invalid molmethod: %s" % str(opts.molmethod))
    sys.exit()
if opts.electrodemethod not in VMDMETHODS:
    print(Fore.RED+"Invalid electrodemethod: %s" % str(opts.electrodemethod))
    sys.exit()

# Loop through input files and process
for fn in opts.infiles:
    ORBS = OrderedDict()
    PROG = getprog(fn)
    if PROG is None:
        print(Fore.RED+Style.BRIGHT+"I don't know what kind of file this is.")
        continue
    print(Back.BLUE+Fore.WHITE+"# # # # # # # # %s (%s) # # # # # # # #" % (fn,PROG))
    BN = os.path.basename(fn)[:-4]
    if PROG == 'xyz':
        tclfn = fn[:-4]+'_vmd.tcl'
        writeSimpleVMD(tclfn, os.path.basename(fn))
        if opts.render and opts.VMDpath:
            print(Fore.BLUE+Back.WHITE+'# # # # # # # # Render  # # # # # # # # # # # #')
            subprocess.run([opts.VMDpath, '-e', tclfn], env=ENV)

    elif PROG == 'cube':
        tclfn = fn[:-5]+'_vmd.tcl'
        writeCubeVMD(tclfn, os.path.basename(fn))

        if opts.render and opts.VMDpath:
            subprocess.run([opts.VMDpath, '-e', tclfn], env=ENV)

    elif PROG == 'orca':
        doorcaprog(fn, opts)
    writePymol(fn[:-4])
