#!/usr/bin/env python3

import sys,os
from parse.orca import *
from output import xyz
import subprocess


class Writer(xyz.Writer):
    def write(self):
        if self.opts.informat == 'gaussian':
            self.__g09_2unform()
            self.__writetransport()
        
    def __g09_2unform(self):
        if subprocess.run(['which', 'g09_2unform'],stdout=subprocess.PIPE).returncode != 0:
            self.logger.error("g09_2unform needs to be in your PATH to convert gaussian outputs to artaios inputs.")
            return None 
        self.logger.info('Writing hamiltonian/overlap: %s' % os.path.split(self.parser.fn)[0] )
        p = subprocess.run(['g09_2unform',self.parser.fn,'1',os.path.split(self.parser.fn)[0]],stdout=subprocess.PIPE).stdout
        for l in str(p,encoding='utf-8').split('\n'):
            if 'reading' in l:
                self.logger.info(l)

    def __guesseletrodes(self):
        '''Try to guess electrodes for transport.in.
           only works if molecule crosses 0 along Z.'''
        e1,mol,e2 = (0,0),(0,0),(0,0)
        self.logger.warn('Assuming atoms are sorted along Z.')
        for atom in ('Au','Ag','S'):
            if atom not in self.parser.zmat.atoms.get_values():
                self.logger.debug('No %s electrodes.' % atom)
                continue
            else:
                self.logger.info('Guessing %s electrodes.' % atom)
            e1 = self.parser.zmat.atoms[self.parser.zmat.atoms == atom][self.parser.zmat.z < 0].index
            e2 = self.parser.zmat.atoms[self.parser.zmat.atoms == atom][self.parser.zmat.z > 0].index
            mol = self.parser.zmat.atoms[self.parser.zmat.atoms != atom].index
        return (e1[0]+1,e1[-1]+1),(mol[0]+1,mol[-1]+1),(e2[0]+1,e2[-1]+1)

    def __writetransport(self):
        self.logger.info('Writing transport.in')
        e1,mol,e2 = self.__guesseletrodes()
        with open(os.path.join(os.path.split(self.parser.fn)[0],'transport.in'), 'w') as fh:
            fh.write('# Total atoms: %i\n' % len(self.parser.zmat))
            fh.write('# Check partitioning for accuracy!\n')
            fh.write('$partitioning\n')
            fh.write('  leftatoms %i-%i\n' % e1) 
            fh.write('  centralatoms %i-%i\n' % mol) 
            fh.write('  rightatoms %i-%i\n' % e2) 
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

def wrblock(fh,l):
    l[-1] = l[-1]+"\n"
    fh.write("\n".join(l))

def countiorb(orbs,nb):
    norb = 0
    for a in orbs:
        norb+= len(orbs[a])
    print("Total iorbs: %s" % norb)
    if nb != norb:
        print("Error: mismatch in number of orbtials and fock matrix")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        print("I need an input file!")
        sys.exit()
    if not os.path.isfile(fn):
        print("I can't read %s" % fn)
        sys.exit()
    with open(fn, 'rt') as fh:
        orbs,orbidx = norbs(fh)
        fm = fock(fh)
        ol = overlap(fh)
    nb = fm.shape[0]
    countiorb(orbs,nb)
    with open('hamiltonian.1', 'wt') as fh:
        l = 0
        for i in range(0,nb):
            for n in range(0,nb):
                fh.write("%s\t" % fm[i][n])
                l += 1
                if l >= 4:
                    fh.write("\n")
                    l = 0
    with open('overlap', 'wt') as fh:
        l = 0
        for i in range(0,nb):
            for n in range(0,nb):
                fh.write("%s " % ol[i][n])
                l += 1
                if l >= 4:
                    fh.write("\n")
                    l = 0

    guessorbs = {"L":[0,0], "M":[0,0], "R":[0,0]}
    with open('transport.in', 'tw') as fh:
        fh.write("#Here is a list of orbitals on each atom:\n")
        i=1
        for a in orbidx:
            ie = i+len(orbs[a])-1
            fh.write("#%s %s-%s\n" % (a, i, ie) )
            if 'Au' in a:
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
            i += len(orbs[a])

        fh.write("##############\n")
        wrblock(fh,["$partitioning","   totnbas  %s" % nb, "   leftbas %s-%s #EDIT THIS!" % tuple(guessorbs["L"]),\
                "   centralbas %s-%s #EDIT THIS!" % tuple(guessorbs["M"]), \
                "   rightbas %s-%s #EDIT THIS!" % tuple(guessorbs["R"]),"$end"])
        wrblock(fh,["$energy_range","  units   eV","  start  -8.0","   end     -1.0",\
                "   fermi_level -5.0","   steps 500", "$end"])
        wrblock(fh,["$system","   nspin  1","$end"])
        wrblock(fh,["$electrodes", "   self_energy wbl","   dos_s 0.036","$end"])
        fh.write("ham_conv 27.21\n")
        fh.write("modelham T\n")
        fh.write("qcprog gen\n")
