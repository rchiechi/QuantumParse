from parse import Parser

import sys,os,re
import numpy as np
from collections import OrderedDict

from parse import xyz
import logging
import warnings
from util import elements
from colorama import Fore,Back,Style



warnings.filterwarnings('ignore','.*None.*',FutureWarning)

class Parser(xyz.Parser):
    
    #TODO Orca parsing is a mess
    fm = None
    orbs = None
    orbidx = None
    ol = None
    breaks = ('CARTESIAN COORDINATES (A.U.)')
    begin = ('CARTESIAN COORDINATES (ANGSTROEM)')

    def parseZmatrix(self):
        self._parsezmat()
        if self.fn[-4:].lower() == '.out':
            self.logger.debug('Parsing overlap and fock matrix from %s' % self.fn)
            with open(self.fn) as fh:
                self.fm = fock(fh)
                self.orbs,self.orbidx = norbs(fh)
                self.ol = overlap(fh)
            if None in (self.fm, self.orbs, self.orbidx, self.ol):
                self.logger.error("Did not parse Orca matrix correctly.")
        else:
            if self.opts.build:
                self.zmat.buildElectrodes(self.opts.build,self.opts.size,
                    self.opts.distance,self.opts.binding,self.opts.surface,self.opts.adatom,self.opts.SAM)
        self.zmat.findElectrodes()

logger = logging.getLogger('OrcaMatrix')

def overlap(fh):
    print("Parsing overlap matrix...")
    fh.seek(0)
    ovdict = OrderedDict()
    inoverlap = False
    lk = []
    for l in fh:
        lk.append(l.strip())
        if not inoverlap:
            if len(lk) > 3: lk.pop(0)
            else: continue
        if lk[1] == 'OVERLAP MATRIX' and not inoverlap:
            inoverlap = True
            continue
        elif l[0] == '-' and inoverlap: break
        if inoverlap:
            lsf = l.split()
            fl = []
            for n in lsf:
                if '.' in n: fl.append(float(n))
                else: fl.append(int(n))
            if isinstance(fl[-1],float):
                if fl[0] not in ovdict: ovdict[fl[0]] = fl[1:]
                else: ovdict[fl[0]] += fl[1:]
    print("%sOverlap Matrix " % Fore.YELLOW, end='')
    print("%sx-elements: %s%s, %sy-elements: %s%s%s" \
            % (Fore.YELLOW,Fore.GREEN,len(ovdict),Fore.YELLOW,Fore.GREEN,len(ovdict[0]),Style.RESET_ALL) )
    return np.array(list(ovdict.values()),np.float)

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
        if key in l: scfidx += 1
    print(Fore.GREEN+str(scfidx)+Style.RESET_ALL)
    fh.seek(0)
    iscf = 0 
    print("Parsing Fock matrix...")
    sys.stdout.flush()
    converged = False
    for l in fh:
        l = l.strip()
        if not l:
            continue
        if 'SCF CONVERGED' in l:
            converged = True
        if 'ERROR' in l and not converged:
            logger.warning('Error detected in Orca output!')
        if key in l:
            iscf += 1
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
            lsf = l.split()
            fl = []
            for n in lsf:
                if '.' in n:
                    try:
                        n = float(n)
                        if n == 0: n = abs(n)
                        fl.append(n)
                    except ValueError as msg:
                        logger.warning('Error parsing fock matrix: %s' %str(msg))
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
                    logger.warning("Out-of-order orbital: %s <= %s" % (norb[-1],norb[-2]))
                    continue
                i = fl[0]
                if i not in fockdict:
                    fockdict[i] = fl[1:]
                else:
                    fockdict[i] += fl[1:]
    fockmatrix = []
    for i in sorted(fockdict.keys()):
        if len(fockdict[i]) != len(fockdict):
            logger.error("Matrix alignment error: {%s} " % i, end='')
            sys.exit()
        else:
            fockmatrix.append(fockdict[i])
    fm = np.array(fockmatrix, np.float)
    if 0 in fm.shape:
        logger.error("Empty Fock Matrix! Shape: %s" % str(fm.shape) )
        sys.exit()
    print("%sFock matrix " % Fore.YELLOW,end='')
    print("%sx-elements: %s%s, %sy-elements: %s%s%s" \
            % (Fore.YELLOW,Fore.GREEN,fm.shape[0],Fore.YELLOW,Fore.GREEN,fm.shape[1],Style.RESET_ALL) )
    if not converged:
        logger.warning('Orca calculation may not have converged!')
    return fm

def norbs(fh):
    fh.seek(0)
    orbdict = OrderedDict()
    inorb = False
    orbidx = []
    lk = []
    rp = re.compile('^\d+\D+$')
    lidx = 0
    logger.info("Parsing molecular orbitals...")
    for l in fh:
        lidx += 1
#        if not lidx%1000000:
#            sys.stdout.write(".")
#            sys.stdout.flush()
        if not inorb:
            lk.append(l.strip())
            if len(lk) > 3: lk.pop(0)
            else: continue
        if lk[1] == 'MOLECULAR ORBITALS' and not inorb: inorb = True
        elif l[0] == '*' and inorb: break
        if inorb:
            lsf = l.split()
            if not lsf: continue
            elif re.match(rp,lsf[0]) == None: continue
            if lsf[0] in orbdict:
                if lsf[1] in orbdict[lsf[0]]: continue
                else: orbdict[lsf[0]].append(lsf[1])
            else:
                orbdict[lsf[0]] = [lsf[1]]
                orbidx.append(lsf[0])
#    print('.')
    torbs = 0
    for a in orbdict: torbs+=len(orbdict[a])
    if torbs == 0:
        logger.error("No orbitals found, caclulation probably did not converge!%s")
        sys.exit()
    else:
        print("%sAtoms: %s%s %sOrbitals: %s%s%s" \
                % (Fore.YELLOW,Fore.CYAN,len(orbdict),Fore.YELLOW,Fore.GREEN,torbs,Style.RESET_ALL) )
    return orbdict, orbidx
