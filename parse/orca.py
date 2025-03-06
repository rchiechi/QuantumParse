import sys
import re
import numpy as np
from collections import OrderedDict
from pathlib import Path
from parse import xyz
import logging
import warnings
# from util import elements
from colorama import Fore, Style

warnings.filterwarnings('ignore','.*None.*',FutureWarning)

class InMatrix:
    
    def __init__(self, insig:int, outsig:int):
        self.active = False
        self.outsig = [False for i in range(outsig)]
        self.insig = [False for i in range(insig)]
    
    def init(self):
        self.active = True
        for i, _ in enumerate(self.insig):
            self.insig[i] = False
        self.pos()
    
    def sig(self):
        if not self.inmatrix:
            self.pos()
        else:
            self.neg()
        return self.inmatrix
    
    def pos(self):
        self._checksig(self.insig)
        return self.inmatrix
    
    def neg(self):
        self._checksig(self.outsig)
        return self.inmatrix
    
    def _checksig(self, sig):
        if not self.active:
            return
        if all(sig):
            raise ValueError("Called in/out in wrong order or too many times.")
        for i, s in enumerate(sig):
            if not s:
                sig[i] = True
                return

    @property
    def inmatrix(self):
        if not self.active:
            raise ValueError("Called inmatrix on unititialized InMatrix.")
        return all(self.insig) and not all(self.outsig)
        

class Parser(xyz.Parser):

    # TODO: Orca parsing is a mess
    fm = None
    orbs = None
    orbidx = None
    ol = None
    breaks = ('CARTESIAN COORDINATES (A.U.)')
    begin = ('CARTESIAN COORDINATES (ANGSTROEM)')


    def __dotransport(self):
        self.logger.debug('Parsing overlap and fock matrix from %s' % self.fn)
        orca_out = []
        # TODO: Deal with unrestricted calculations
        
        with Path(self.fn).open() as fh:
            for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            orca_out.append(parts)
        self.ol = overlap(orca_out)
        self.fm = fock(orca_out)
        if self.opts.unrestricted:
            self.logger.debug("Parsing unrestricted calculation")
            self.fm_beta = fock(orca_out, 1)
        else:
            self.fm_beta = None
        self.orbs,self.orbidx = norbs(orca_out)
            
        if 0 in (len(self.fm), len(self.orbs), len(self.orbidx), len(self.ol)):
            self.logger.error("Did not parse Orca matrix correctly.")


logger = logging.getLogger('OrcaMatrix')

def overlap(orca_out):
    print("Parsing overlap matrix...")
    inoverlap = InMatrix(2,1)
    matrix_data = []
    orb_idx = -1
    for parts in orca_out:
        if "OVERLAP MATRIX" in " ".join(parts):
            inoverlap.init()
            continue
        elif not inoverlap.active:
            continue
        if not parts[0].isdigit():
            if not inoverlap.sig():
                break
            continue
        if not all(val.isdigit() for val in parts):
            if int(parts[0]) > orb_idx:
                orb_idx = int(parts[0])
            elif int(parts[0]) == 0:
                orb_idx = 0
            else:
                break
            if len(matrix_data) <= orb_idx:
                matrix_data.append(parts[1:])
            else:
                matrix_data[orb_idx] += parts[1:]
    om = np.array(matrix_data, float)
    if 0 in om.shape or om.shape[0] != om.shape[1]:
        logger.error("Bad overlap matrix! Shape: %s", str(om.shape))
        sys.exit()
    print("%Overlap matrix " % Fore.YELLOW,end='')
    print("%sx-elements: %s%s, %sy-elements: %s%s%s" % (Fore.YELLOW,Fore.GREEN,
                                                        om.shape[0],Fore.YELLOW,
                                                        Fore.GREEN,om.shape[1],
                                                        Style.RESET_ALL))  
    return  om

def fock(orca_out, spin=0):
    key = f"Fock matrix for operator {spin}"
    # endkey = f"Fock matrix for operator {abs(spin-1)}"
    fock_idx = -1
    matrix_data = []
    infock = InMatrix(1,1)
    scfidx = 0
    nscf = -1
    print("Finding last Fock matrix...", end=' ', flush=True)
    for idx, parts in enumeragte(orca_out):
        if "SCF CONVERGED" in " ".join(parts):
            for part in parts:
                if part.isdigit:
                    nscf = int(part)
        if "ERROR" in parts and not nscf:
            logger.warning(f"Error detected in Orca output: {" ".join(parts)}")
        if " ".join(parts) == key:
            scfidx  = idx
    print(f"{Fore.GREEN}SCF Converted in {nscf} cycles{Style.RESET_ALL}")
    print("Parsing Fock matrix...", flush=True)
    infock.init()
    for parts in orca_out[scfidx:]:
        if not parts[0].isdigit():
            if not infock.sig():
                break
            continue
        if not all(val.isdigit() for val in parts):
            if int(parts[0]) > fock_idx:
                fock_idx = int(parts[0])
            elif int(parts[0]) == 0:
                fock_idx = 0
            else:
                break
            if len(matrix_data) <= fock_idx:
                matrix_data.append(parts[1:])
            else:
                matrix_data[fock_idx] += parts[1:]
#         if infock.inmatrix:
#             if _l == '<<< The NR Solver signals convergence >>>':
#                 continue
#             ditch = False
#             lsf = _l.split()
#             fl = []
#             for n in lsf:
#                 if '.' in n:
#                     try:
#                         n = float(n)
#                         if n == 0:
#                             n = abs(n)
#                         fl.append(n)
#                     except ValueError as msg:
#                         logger.warning('Error parsing fock matrix: %s', str(msg))
#                         ditch = True
#                         continue
#                 else:
#                     n = int(n)
#                     fl.append(n)
# 
#             if ditch:
#                 continue
#             if isinstance(fl[-1],float) and isinstance(fl[0],int):
#                 norb.append(fl[0])
#                 if 0 < norb[-1] <= norb[-2]:
#                     logger.warning("Out-of-order orbital: %s <= %s" % (norb[-1],norb[-2]))
#                     continue
#                 i = fl[0]
#                 if i not in fockdict:
#                     fockdict[i] = fl[1:]
#                 else:
#                     fockdict[i] += fl[1:]
#     fockmatrix = []
#     for i in sorted(fockdict.keys()):
#         if len(fockdict[i]) != len(fockdict):
#             logger.error("Matrix alignment error: {%s} " % i, end='')
#             return fockmatrix
#         else:
#             fockmatrix.append(fockdict[i])
    fm = np.array(matrix_data, float)
    if 0 in fm.shape or fm.shape[0] != fm.shape[1]:
        logger.error("Bad Fock matrix! Shape: %s", str(fm.shape))
        sys.exit()
    print("%sFock matrix " % Fore.YELLOW,end='')
    print("%sx-elements: %s%s, %sy-elements: %s%s%s" % (Fore.YELLOW,Fore.GREEN,
                                                        fm.shape[0],Fore.YELLOW,
                                                        Fore.GREEN,fm.shape[1],
                                                        Style.RESET_ALL))  
    return fm




def norbs(ocra_out):
    orbdict = OrderedDict()
    inorb = InMatrix(2,1)
    orbidx = []
    lk = []
    rp = re.compile(r'^\d+\D{1,2}$')
    lidx = 0
    logger.info("Parsing molecular orbitals...")
    for parts in orca_out:
        if "MOLECULAR ORBITALS" in " ".join(parts):
            inoverlap.init()
            continue
        elif not inoverlap.active:
            continue
        if len(parts) == 1 and parts[0][0] == '-':
            inoverlap.sig()
            continue
        if parts[0][0] == '*':
            if not inoverlap.sig():
                break
            continue
        if re.match(rp, parts[0]) is None:
            continue
        # if inorb:
        # lsf = _l.split()
        # if not lsf:
        #     inorb = False
        #     break
        # elif re.match(rp,lsf[0]) is None:
        #     continue
        if parts[0] in orbdict:
            if parts[1] in orbdict[parts[0]]:
                continue
            else:
                orbdict[parts[0]].append(parts[1])
        else:
            orbdict[parts[0]] = [parts[1]]
            orbidx.append(parts[0])
    torbs = 0
    for a in orbdict:
        torbs += len(orbdict[a])
    if torbs == 0:
        logger.error("No orbitals found, caclulation probably did not converge!%s")
        sys.exit()
    else:
        print("%sAtoms: %s%s %sOrbitals: %s%s%s" % (Fore.YELLOW,Fore.CYAN,len(orbdict),
                                                    Fore.YELLOW,Fore.GREEN,torbs,Style.RESET_ALL))
    return orbdict, orbidx
