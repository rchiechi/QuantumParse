#!/usr/bin/env python3

import sys
import os
import re
import asyncio
import stat
import shutil
import argparse
from collections import OrderedDict
from pathlib import Path

## Artaios binary
BIN = os.environ.get('ARTAIOS', os.path.join(os.path.expanduser('~'),"source/artaios/bin/artaios"))
##


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel Artaios Transport Calculation')
    
    # Input file argument
    parser.add_argument('input_file', 
                        nargs='?', 
                        default='transport.in',
                        help='Input file for transport calculation (default: transport.in)')
    
    # Optional arguments can be added here
    parser.add_argument('-n', '--nthreads', 
                        type=int, 
                        help='Number of simltanous jobs to use (overrides automatic detection)')
    
    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Enable verbose output')

    # Parse arguments
    args = parser.parse_args()

    return args


######### Main () ############################

async def main():
    
    args = parse_arguments()
    
    IN = Path(args.input_file)
    
    # Optional: override NCPU if specified
    if args.ncpu:
        NCPU = args.ncpu
    else:
        try:
            from psutil import cpu_count
            NCPU=cpu_count()
        except ImportError:
            print('You need to install psutil.')
            print('e.g., sudo -H pip3 install --upgrade psutil')
            sys.exit(1)
    
    if shutil.which('parallel') is None:
        print("Error: GNU Parallel is not installed.")
        print("Please install it using one of these methods:")
        print("  - On Ubuntu/Debian: sudo apt-get install parallel")
        print("  - On Fedora: sudo dnf install parallel")
        print("  - On macOS with Homebrew: brew install parallel")
        print("  - On macOS with MacPorts: sudo port install parallel")
        print("  - With pip: pip install GNU-parallel")
        sys.exit(1)
    
    TDIR = Path('artaios_parallel')
    INPUTS= ('hamiltonian.1', 'overlap')
    BFILE = Path('artaios_parallel.sh')
    
    while not IN.exists():
        for fn in os.listdir():
            if 'transport.in' in fn:
                print(f"Guessing input as {fn}")
                IN = Path(fn)
        break
    
    if not IN.exists():
        print(f"{IN} does not exist.")
        sys.exit()
    
    transport = IN.read_text()
    
    rbas = ''
    xyzfile = ''
    fermi_level = ''
    mosfile = ''
    # electrodes = []
    transport = []
    energy = {"start": None,
            "end": None,
            "steps": None}
    
    res = re.compile(r'\s*(\w+)\s+(-?\d)', re.I)
    
    with open(IN, 'r') as fh:
        inrange=False
        inelectrodes=False
        for l in fh:
            if "rbas" in l.lower():
                rbas = l.strip().split(' ')[-1]
            if "xyzfile" in l.lower():
                xyzfile = l.strip().split(' ')[-1]
            if "mosfile" in l.lower():
                #No spaces in file names!
                mosfile = l.strip().split(' ')[-1]
            if "fermi_level" in l.lower():
                fermi_level = l.strip().split(' ')[-1]
                #Do not write this line (we need Ef between energy ranges)
                # continue
            if "$energy_range" in l.lower():
                inrange = True
                continue
            elif "$end" in l.lower() and inrange:
                inrange = False
                continue
            if inrange:
                m = re.search(r'\s*(\w+)\s+(-?\d+)', l.strip(), re.I)
                if m:
                    try:
                        k, v = m.groups()
                        float(v)
                        if k.lower() not in energy:
                            raise ValueError(f"{k} not in energy dict")
                        energy[k.lower()] = v.strip()
                    except ValueError as msg:
                        print(f"Error finding value in {l.strip()}: {msg}")
                        sys.exit()
            else:
                transport.append(l.strip())
    
    energy['steps'] = int(energy['steps'])
    energy['start'] = float(energy['start'])
    energy['end'] = float(energy['end'])
    
    for e in energy:
        print("%s: %s" % (e, energy[e]))
    remainder = energy['steps']%NCPU
    
    print("Check the math...")
    print("Steps/nCPU: %s/%s = %s (%s)" % (energy['steps'],NCPU,int(energy['steps']/NCPU),remainder) )
    
    jobs = OrderedDict()
    for i in range(0,NCPU):
        jobs[i]=int((energy['steps']-remainder)/NCPU)
    i = 0
    while remainder:
        jobs[i] = jobs[i]+1
        remainder -=1
        i += 1
        if i not in jobs:
            i = 0
    
    s = 0
    for j in jobs:
        s += jobs[j]
    
    print("Steps: %s = %s" % (s,energy['steps']) )
    print("Interval: %s - %s = %s" % (energy['end'],energy['start'], energy['end']-energy['start']) )
    
    interval = energy['end']-energy['start']
    mstep = interval/energy['steps']
    
    print("Interval/steps: %s / %s = %s (%s)" % (interval,energy['steps'],interval/energy['steps'],interval%energy['steps'] ) )
    
    s = energy['start']
    e = energy['start']
    for j in jobs:
        e += jobs[j]*mstep
        jobs[j] = (jobs[j],s,e)
        s = e
    
    print("End: %0.2f = %0.2f" % (s,energy['end']), end='\n\n')
    
    for j in jobs:
        #print('Thread %s: %.4f -> %.4f ()' % (j+1, jobs[j][1], jobs[j][2]))
        print(f"Thread {j+1}: {jobs[j][1]:.4f} -> {jobs[j][2]:.4f} Steps: {jobs[j][0]}")
    
    
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * ")
    print(" Creating directories under %s " % TDIR)
    print("* * * * * * * * * * * * * * * * * * * * * * * * ")
    
    if not TDIR.exists():
        os.mkdir(TDIR)
    else:
        print("Cleanup %s fist" % TDIR)
        sys.exit()
    
    BFILE.write_text("#!/bin/bash\n\n" +
                    f"ARTAIOS={BIN}\n\n" +
                    f"LOG={os.environ['PWD']}/artaios.log\n\n" +
                    f"parallel -j {NCPU} <<EOF\n")
    
    for j in jobs:
        idx = str(j)
        (TDIR / idx).mkdir()
        for I in INPUTS:
            if not os.path.exists(I):
                print(f"{I} does not exist!")
                sys.exit()
            os.symlink(f'../../{I}', TDIR / idx / I)
        if mosfile:
            os.symlink(f'../../{mosfile}', TDIR / idx / mosfile)
        if xyzfile:
            os.symlink(f'../../{xyzfile}', TDIR / idx / xyzfile)
        if rbas:
        os.symlink(f'../../{rbas}', TDIR / idx / rbas)
        energy_range = "$energy_range\n"
        for e in energy:
            if e.lower() not in ('start','end','steps'):
                energy_range+=f"  {e} {energy[e]}\n"
            elif e.lower() ==  'steps':
                energy_range += f"  {e} {jobs[j][0]}\n"
            elif e.lower() ==  'start':
                energy_range+=f"  {e} {jobs[j][1]:4f}\n"
            elif e.lower() ==  'end':
                energy_range+=f"  {e} {jobs[j][2]:4f}\n"
        energy_range+="$end\n"
        t_in = TDIR / idx / 'transport.in'
        t_in.write_text("\n".join(transport) +
                    #  "$electrodes\n" +
                    #  "".join(electrodes) +
                    # f"  fermi_level {jobs[j][1]}\n" +
                    # #Set fermi_level to the lower energy range to avoid an error in artaios
                    # "  fermi_level %s\n" % jobs[j][1] +
                    # "$end\n" +
                    energy_range)
        with BFILE.open('a') as fh:
            fh.write(f"cd {TDIR / idx}; " +
                    "$ARTAIOS transport.in | tee artaios.log\n")
    
    with BFILE.open('a') as fh:
        fh.write("EOF\n")
        fh.write('find "%s" -name transmission.1.dat -print0 | xargs -0I {} cat {} >> transmission.1.dat\n' % TDIR)
        fh.write('gnuplot transmission.gpin\n')
    
    Path('transmission.gpin').write_text('set term postscript color\n'+
                                        'set term pngcairo\n'+
                                        'set output "transmission.png"\n'+
                                        'set nokey\n'+
                                        'set title "Transmission"\n'+
                                        'set xlabel "E-E_f (eV)"\n'+
                                        'set ylabel "transmission"\n'+
                                        'set logscale y\n'+
                                        f'plot "transmission.1.dat"  u ($1-{fermi_level}):2 w l smooth unique\n')
    
    BFILE.chmod(BFILE.stat().st_mode | stat.S_IEXEC)
    
