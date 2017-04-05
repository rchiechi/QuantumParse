#!/usr/bin/env python3

import sys,os,stat
from collections import OrderedDict
from psutil import cpu_count

BIN = os.path.join(os.path.expanduser('~'),"source/artaios-030417/bin/artaios")

NCPU=cpu_count()
TDIR = 'artaios_parallel'
INPUTS= ('hamiltonian.1', 'overlap')
BFILE = 'artaios_parallel.sh'

if not len(sys.argv) > 1 and not os.path.exists('transport.in'):
    print("I need an input file")
    sys.exit()
elif os.path.exists('transport.in'):
    IN='transport.in'
else:
    IN=sys.argv[1]

mosfile = ''
transport = []
energy = OrderedDict()
with open(IN, 'r') as fh:
    inrange=False
    for l in fh:
        if "mosfile" in l.lower():
            #No spaces in file names!
            mosfile = l.strip().split(' ')[-1]
        if "$energy_range" in l.lower():
            inrange = True
            continue
        elif "$end" in l.lower() and inrange:
            inrange = False
            continue
        if inrange:
            kv = []
            for _l in filter(None,l.strip().split(' ')):
                kv.append(_l)
            try:
                energy[kv[0]]=float(kv[1])
            except:
                energy[kv[0]]=kv[1]
        else:
            transport.append(l)
try:
    energy['steps'] = int(energy['steps'])
except KeyError:
    print('Error finding steps.')
    sys.exit()

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
    print('Thread %s: %.4f -> %.4f' % (j+1, jobs[j][1], jobs[j][2]))


print("\n* * * * * * * * * * * * * * * * * * * * * * * * ")
print(" Creating directories under %s " % TDIR)
print("* * * * * * * * * * * * * * * * * * * * * * * * ")

if not os.path.exists(TDIR):
    os.mkdir(TDIR)
else:
    print("Cleanup %s fist" % TDIR)
    sys.exit()

with open(BFILE, 'w') as fh:
    fh.write("#!/bin/bash\n\nARTAIOS=%s\n\nLOG=%s/artaios.log\n\n" % (BIN,os.environ['PWD']))

for j in jobs:
    os.mkdir(os.path.join(TDIR,str(j)))
    for I in INPUTS:
        if not os.path.exists(I):
            print("%s does not exist!" % I)
            sys.exit()
        os.symlink('../../%s' % I, os.path.join(TDIR,str(j),I))
    if mosfile:
        os.symlink('../../%s' % mosfile, os.path.join(TDIR,str(j),mosfile))
    energy_range="$energy_range\n"
    for e in energy:
        if e.lower() not in ('start','end','steps'):
            energy_range+="  %s %s\n" % (e,energy[e])
        elif e.lower() ==  'steps':
            energy_range+="  %s %s\n" % (e,jobs[j][0])
        elif e.lower() ==  'start':
            energy_range+="  %s %.4f\n" % (e,jobs[j][1])
        elif e.lower() ==  'end':
            energy_range+="  %s %.4f\n" % (e,jobs[j][2])
    energy_range+="$end\n"
    with open(os.path.join(TDIR,str(j),'transport.in'), 'w') as fh:
        fh.write("".join(transport))
        fh.write(energy_range)
    with open(BFILE, 'a') as fh:
        fh.write("(cd %s; $ARTAIOS transport.in | tee artaios.log)&\n" % os.path.join(TDIR,str(j)) )

with open(BFILE, 'a') as fh:
    fh.write("wait\n")
    fh.write('find "%s" -name transmission.1.dat -print0 | xargs -0I {} cat {} >> transmission.1.dat\n' % TDIR)
    fh.write('gnuplot transmission.gpin\n')

with open('transmission.gpin', 'wt') as fh:
    fh.write('set term postscript color\n')
    fh.write('set term pngcairo\n')
    fh.write('set output "transmission.png"\n')
    fh.write('set nokey\n')
    fh.write('set title "Transmission"\n')
    fh.write('set xlabel "E-E_f (eV)"\n')
    #fh.write('set xrange [%f:%f]\n' % (energy['start'],energy['end']))
    fh.write('set ylabel "transmission"\n')
    fh.write('set logscale y\n')
    fh.write('plot "transmission.1.dat"  u ($1-%s):2 w l smooth unique\n' % -5.0)

os.chmod(BFILE, os.stat(BFILE).st_mode | stat.S_IEXEC)
