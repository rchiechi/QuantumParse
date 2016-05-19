#!/usr/bin/env python3

import sys,os
from parse.orca import *
from output import xyz
import subprocess
import numpy as np
from util import *

class Writer(xyz.Writer):
    #TODO Orca is a mess
    def write(self):
        if self.opts.informat == 'gaussian':
            self.__g09_2unform()
            self.__writetransport()
        elif self.opts.informat == 'orca':
            self.parser.parseMatrix()
            self.writeOrcatransport()

    def __g09_2unform(self):
        if subprocess.run(['which', 'g09_2unform'],stdout=subprocess.PIPE).returncode != 0:
            self.logger.error("g09_2unform needs to be in your PATH to convert gaussian outputs to artaios inputs.")
            return None 
        self.logger.info('Writing hamiltonian/overlap: %s' % os.path.split(self.parser.fn)[0] )
        p = subprocess.run(['g09_2unform',self.parser.fn,'1',os.path.split(self.parser.fn)[0]],stdout=subprocess.PIPE)
        for l in str(p.stdout,encoding='utf-8').split('\n'):
            if 'reading' in l:
                self.logger.info(l)

    def __writetransport(self):
        if not self.parser.haselectrodes():
            self.logger.error('Did not parse any electrodes.')
            return
        self.logger.info('Writing transport.in')
        fp = os.path.join(os.path.split(self.parser.fn)[0],'transport.in')
        if os.path.exists(fp) and not self.opts.overwrite:
            self.logger.error('Not overwriting %s' %fp)
            return
            
        #with open(os.path.join(os.path.split(self.parser.fn)[0],'transport.in'), 'w') as fh:
        with open(fp, 'w') as fh:
            fh.write('# Total atoms: %i\n' % len(self.parser.zmat))
            fh.write('# Guessed electrodes as %s\n' % self.parser.zmat.electrodes['atom'])
            fh.write('# Check partitioning for accuracy!\n')
            fh.write('$partitioning\n')
            fh.write('  leftatoms %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['L'])+1)) 
            fh.write('  centralatoms %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['M'])+1)) 
            fh.write('  rightatoms %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['R'])+1)) 
            fh.write('$end\n')
            fh.write('$energy_range\n')
            fh.write('  units   eV\n')
            fh.write('  start  -8.0\n')
            fh.write('  end     -1.0\n')
            fh.write('  fermi_level -5.0\n')
            fh.write('  steps 200\n')
            fh.write('$end\n')
            fh.write('$system\n')
            fh.write('  nspin  %i\n' % xyz.Writer.getMultiplicity(self.parser.zmat))
            fh.write('$end\n')
            fh.write('$electrodes\n')
            fh.write('  self_energy wbl\n')
            fh.write('  dos_s 0.036\n')
            fh.write('$end\n\n')
            fh.write('ham_conv 27.21\n')
            fh.write('modelham F\n')
            fh.write('rbas T\n')
            fh.write('qcprog g09\n')
            fh.write('mosfile %s\n' % os.path.basename(self.parser.fn))
            fh.write('bondflux T\n')
            fh.write('loewdin_central T\n')
            fh.write('fluxdir Z\n')
            fh.write('fluxsurf -5.0 5.0\n')
            fh.write('fluxthres 0.3\n')

    def __wrblock(self,fh,l):
        l[-1] = l[-1]+"\n"
        fh.write("\n".join(l))

    def __countiorb(self,nb):
        norb = 0
        for a in self.parser.orbs:
            norb+= len(self.parser.orbs[a])
        print("Total iorbs: %s" % norb)
        if nb != norb:
            print("Error: mismatch in number of orbtials and fock matrix")

    def writeOrcatransport(self):
        
        nb = self.parser.fm.shape[0]
        self.__countiorb(nb)
        with open('hamiltonian.1', 'wt') as fh:
            l = 0
            for i in range(0,nb):
                for n in range(0,nb):
                    fh.write("%s\t" % self.parser.fm[i][n])
                    l += 1
                    if l >= 4:
                        fh.write("\n")
                        l = 0
        with open('overlap', 'wt') as fh:
            l = 0
            for i in range(0,nb):
                for n in range(0,nb):
                    fh.write("%s " % self.parser.ol[i][n])
                    l += 1
                    if l >= 4:
                        fh.write("\n")
                        l = 0

        guessorbs = {"L":[0,0], "M":[0,0], "R":[0,0]}
        with open('transport.in', 'tw') as fh:
            fh.write("#Here is a list of orbitals on each atom:\n")
            i=1
            for a in self.parser.orbidx:
                ie = i+len(self.parser.orbs[a])-1
                fh.write("#%s %s-%s\n" % (a, i, ie) )
                if a in EATOMS:
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
                i += len(self.parser.orbs[a])

            fh.write("##############\n")
            self.__wrblock(fh,["$partitioning","   totnbas  %s" % nb, "   leftbas %s-%s #CHECK THIS!" % tuple(guessorbs["L"]),\
                    "   centralbas %s-%s #CHECK THIS!" % tuple(guessorbs["M"]), \
                    "   rightbas %s-%s #CHECK THIS!" % tuple(guessorbs["R"]),"$end"])
            self.__wrblock(fh,["$energy_range","  units   eV","  start  -8.0","   end     -1.0",\
                    "   fermi_level -5.0","   steps 500", "$end"])
            self.__wrblock(fh,["$system","   nspin  1","$end"])
            self.__wrblock(fh,["$electrodes", "   self_energy wbl","   dos_s 0.036","$end"])
            fh.write("ham_conv 27.21\n")
            fh.write("modelham T\n")
            fh.write("qcprog gen\n")
