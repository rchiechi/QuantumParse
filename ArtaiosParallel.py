#!/usr/bin/env python3

import sys,os,stat

NCPU=8
BIN = os.path.join(os.environ['HOME'],"source/artaios_beta_020914/bin/artaios")

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

transport = []
energy = {}
with open(IN, 'r') as fh:
    inrange=False
    for l in fh:
        if "$energy_range" in l.lower():
            inrange = True
            continue
        elif "$end" in l.lower():
            inrange = False
            transport.append(l)
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
for e in energy:
    print("%s: %s" % (e, energy[e]))
remainder = energy['steps']%NCPU

print("Check the math...")
print("%s/%s = %s (%s)" % (energy['steps'],NCPU,int(energy['steps']/NCPU),energy['steps']%NCPU) )

jobs = {}
for i in range(0,NCPU):
    jobs[i]=[int((energy['steps']-remainder)/NCPU)]
i = 0
while remainder:
    jobs[i][0] = jobs[i][0]+1
    remainder -=1
    i += 1
    if i not in jobs:
        i = 0

s = 0
for j in jobs:
    s += jobs[j][0]
if s != energy['steps']:
    print("Error distributing jobs")
    sys.exit()

print("%s = %s" % (s,energy['steps']) )
print("%s - %s = %s" % (energy['end'],energy['start'], energy['end']-energy['start']) )

interval = energy['end']-energy['start']
mstep = interval/energy['steps']

print("%s / %s = %s (%s)" % (interval,energy['steps'],interval/energy['steps'], interval%energy['steps'] ) )

s = energy['start']
e = energy['start']
for j in jobs:
    e += jobs[j][0]*mstep
    jobs[j] = (jobs[j][0],s,e)
    s = e

print("%s =? %s" % (int(s),interval))

for j in jobs:
    print('%.4f -> %.4f' % jobs[j][1:])


print("\n * * * * * * * * * * * * * * * * \n Creating directories under %s \n * * * * * * * * * * * * * * * * \n" % TDIR)

if not os.path.exists(TDIR):
    os.mkdir(TDIR)
else:
    print("Please cleanup %s fist" % TDIR)
    sys.exit()

with open(BFILE, 'w') as fh:
    fh.write("#!/bin/bash\n\nARTAIOS=%s\n\nLOG=%s/artaios.log\n\n" % (BIN,os.environ['PWD']))

for j in jobs:
    os.mkdir(os.path.join(TDIR,str(j)))
    for I in INPUTS:
        if not os.path.exists(I):
            print("%s does not exist!" % I)
            sys.exit()
        os.symlink('../../%s'%I, os.path.join(TDIR,str(j),I))

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
        fh.write("(cd %s; $ARTAIOS)&\n" % os.path.join(TDIR,str(j)) )

with open(BFILE, 'a') as fh:
    #fh.write("sleep 5")
    #fh.write("tail -f $LOG")
    fh.write("wait\n")
    fh.write('find "%s" -name transmission.1.dat -print0 | xargs -0I {} cat {} >> transmission.1.dat\n' % TDIR)
os.chmod(BFILE, os.stat(BFILE).st_mode | stat.S_IEXEC)
