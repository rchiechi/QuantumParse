#!/usr/bin/env python3

import sys,os
from parse.orca import *
from output import xyz
import subprocess
import numpy as np
from util import *
#import fortranformat as ff

class Writer(xyz.Writer):
    #TODO Orca is a mess

    def write(self):
        self.spin = 1
        if self.opts.unrestricted:
            self.logger.warn("Unrestricted calculation.")
            self.spin = 2
        self.transport = self.jobname+'.transport.in'
        self.logger.debug('Writing to %s' % self.transport)
        if self.opts.informat == 'gaussian':
            self.__g09_2unform()
            self.__writetransport()
        elif self.opts.informat == 'adf':
            self.__adf_2unform()
            self.__writetransport()
        elif self.opts.informat == 'orca':
            #self.parser.parseMatrix()
            self.writeOrcatransport()

    def __g09_2unform(self):
        if subprocess.run(['which', 'g09_2unform'],stdout=subprocess.PIPE).returncode != 0:
            self.logger.error("g09_2unform needs to be in your PATH to convert gaussian outputs to artaios inputs.")
            self.WriteGaussiantransport()
            return None
        self.logger.info('Writing hamiltonian/overlap: %s' % os.path.split(self.parser.fn)[0] )
        p = subprocess.run(['g09_2unform',self.parser.fn, str(self.spin), os.path.split(self.parser.fn)[0]], stdout=subprocess.PIPE)
        for l in str(p.stdout,encoding='utf-8').split('\n'):
            if 'reading' in l:
                self.logger.info(l)

    def __adf_2unform(self):
        if subprocess.run(['which', 'adf2unform'],stdout=subprocess.PIPE).returncode != 0:
            self.logger.error("adf2unform needs to be in your PATH to convert gaussian outputs to artaios inputs.")
            return None
        self.logger.info('Writing hamiltonian/overlap: %s' % os.path.split(self.parser.fn)[0] )
        p = subprocess.run(['adf2unform',self.parser.fn, str(self.spin), '2014'], stdout=subprocess.PIPE)
        # ADF version 2014 parsing works at least up to ADF 2019.304
        for l in str(p.stdout,encoding='utf-8').split('\n'):
            if 'reading' in l:
                self.logger.info(l)

    def __writetransport(self):
        if not self.parser.haselectrodes():
            self.logger.warn('Did not parse any electrodes.')
            #return
        self.logger.info('Writing %s' % self.transport)
        #fp = os.path.join(os.path.split(self.parser.fn)[0],'transport.in')
        fp = os.path.join(os.path.split(self.parser.fn)[0],self.transport)

        if os.path.exists(fp) and not self.opts.overwrite:
            self.logger.error('Not overwriting %s' %fp)
            return

        with open(fp, 'w') as fh:
            fh.write('# Total atoms: %i\n' % len(self.parser.zmat))
            fh.write('# Guessed electrodes as %s\n' % self.parser.zmat.electrodes['atom'])
            fh.write('# Check partitioning for accuracy!\n')
            fh.write('$partitioning\n')
            fh.write('  leftatoms %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['L'])+1))
            fh.write('  centralatoms %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['M'])+1))
            fh.write('  rightatoms %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['R'])+1))
            fh.write('$end\n\n')
            fh.write('$energy_range\n')
            fh.write('  start  -8.0\n')
            fh.write('  end     -1.0\n')
            fh.write('  steps 200\n')
            fh.write('$end\n\n')
            fh.write('$system\n')
            #if self.opts.unrestricted:
            fh.write('  nspin  %i\n' % self.spin)
            #else:
            #    fh.write('  nspin  %i\n' % xyz.Writer.getMultiplicity(self.parser.zmat))
            fh.write('$end\n\n')
            fh.write('$electrodes\n')
            fh.write('  self_energy wbl\n')
            fh.write('  dos_s 0.036\n')
            fh.write('  fermi_level -5.0\n')
            fh.write('$end\n\n')
            fh.write('$general\n')
            fh.write('  do_transport\n')
            fh.write('  unit   eV\n')
            fh.write('#  conductance\n')
            fh.write('  loewdin_central\n')
            fh.write('  rbas\n')
            if self.opts.informat == 'gaussian':
                fh.write('  qcprog g09\n')
            elif self.opts.informat == 'adf':
                fh.write('  qcprog adf\n')
            fh.write('  mosfile %s\n' % os.path.basename(self.parser.fn))
            fh.write('$end\n\n')
            fh.write('#Uncomment below to enable local bondflux. Check fluxdir!\n')
            fh.write('#$local_transmission\n')
            fh.write('#  bondflux\n')
            fh.write('#  fluxdir Z\n')
            fh.write('#  fluxsurf -5.0 5.0\n')
            fh.write('#  fluxthres 0.3\n')
            fh.write('#  bc_scale_area\n')
            fh.write('#  atomgroup %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['M'])+1))
            fh.write('#  #read_green\n')
            fh.write('#  #print_green\n')
            fh.write('#$end\n\n')
            fh.write('$subsystem\n')
            fh.write('  print_molden T\n')
            fh.write('  do_diag_central T\n')
            fh.write('  print_diag_central T\n')
            fh.write('  moldeninfile %s.molden.input\n' % os.path.splitext(os.path.basename(self.parser.fn))[0])
            fh.write('$end\n\n')

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

    def WriteGaussiantransport(self):
        # import fortranformat as ff
        # a = [ ... , ... ]
        # lineformat = ff.FortranRecordWriter('(4D22.12)')
        #
        # 4 Double precision per line, 22-characters wide, 12 digit precision
        # lineformat.write(a)
        # '    0.100000000000D+01    0.123456789000D+22    0.000000000000D+00    0.100000000000D-11\n     0.100000000000D+01...'
        self.logger.info('Writing transport data using internal parser.')

        def __fortranformat(fn, ar):
            with open(fn, 'wt') as fh:
                self.logger.info('Writing %s...' % fn)
                i = 0
                #for (x,y), value in np.ndenumerate(ar):
                for value in np.nditer(ar):
                    #print('%s,%s' % (x,y), end='\r')
                    if i == 4:
                        i = 0
                        fh.write('\n')
                    if value < 0:
                        sp = '   '
                    else:
                        sp = '    '
                    fh.write( str('%s%.12E' % (sp,value) ).replace('E','D'))
                    i += 1
                if i != 1:
                    fh.write('\n')
        __fortranformat('overlap', self.parser.ol)
        __fortranformat('hamiltonian.1', self.parser.fm)
        if self.opts.unrestricted:
            __fortranformat('hamiltonian.2', self.parser.fm_beta) # WARNING: NOt implemented!
        #with open('hamiltonian.1', 'wt') as fh:
        #    self.logger.info('Writing Hamiltonian...')
        #    i = 0
        #    for (x,y), value in np.ndenumerate(self.parser.fm):
        #        print('%s,%s' % (x,y), end='\r')
        #        if i == 4:
        #            i = 0
        #            fh.write('\n')
        #        if value < 0:
        #            sp = '   '
        #        else:
        #            sp = '    '
        #        fh.write( str('%s%.12E' % (sp,value) ).replace('E','D'))
        #        i += 1
        return
            #fh.write(lineformat.write(np.nditer(self.parser.ol)))
        #for (x,y), value in np.ndenumerate(self.parser.fm):
            #print('%s,%s' % (x,y), end='\r')
            #fock.append(value)
        #with open('hamiltonian.1', 'wt') as fh:
            #self.logger.info('Writing hamiltonian...')
            #fh.write(lineformat.write(fock))

#        with open('hamiltonian.1', 'wt') as fh:
#            self.logger.info('Writing hamiltonian...')
#            i = 0
#            for (x,y), value in np.ndenumerate(self.parser.fm):
#                print('%s,%s' % (x,y), end='\r')
#                fock.append(value)
#                #overlap.append(self.parser.ol[x,y])
#                #fock.append(self.parser.fm[x,y])
#                if i == 4:
#                    i = 0
#                    fock = []
#                    fh.write('\n')
#                fh.write(lineformat.write(fock))

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
        #TODO FIX THIS
        if self.opts.unrestricted:
            nb_beta = self.parser.fm_beta.shape[0]
            self.__countiorb(nb_beta)
            with open('hamiltonian.2', 'wt') as fh:
                l = 0
                for i in range(0,nb_beta):
                    for n in range(0,nb_beta):
                        fh.write("%s\t" % self.parser.fm_beta[i][n])
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
        with open(self.transport, 'tw') as fh:
            fh.write("#Here is a list of orbitals on each atom:\n")
            i=1
            for a in self.parser.orbidx:
                ie = i+len(self.parser.orbs[a])-1
                fh.write("#%s %s-%s\n" % (a, i, ie) )
                _a = ''
                for c in a:
                    try:
                        int(c)
                    except ValueError:
                        _a += c
                if _a == self.parser.zmat.electrodes['atom']:
                    if guessorbs["L"][0] == 0:
                        guessorbs["L"][0] = i
                        guessorbs["L"][1] = ie
                    elif guessorbs["M"] == [0,0]:
                        guessorbs["L"][1] = ie
                    elif guessorbs["R"][0] == 0:
                        guessorbs["R"][0] = i
                        guessorbs["R"][1] = ie
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
            self.__wrblock(fh,["$energy_range","  start  -8.0","   end     -1.0", "   steps 200", "$end"])
            self.__wrblock(fh,["$system","   nspin  %i" % self.spin,"$end"])
            self.__wrblock(fh,["$electrodes", "   self_energy wbl","   dos_s 0.036", '   fermi_level -5.0', "$end"])
            self.__wrblock(fh,["$general", '  do_transport', '  unit   eV', '  modelham', '  loewdin_central', \
                               '  qcprog gen', '#  conductance', '$end'])
            fh.write('#Uncomment below to enable local bondflux. Check fluxdir!\n')
            fh.write('#$local_transmission\n')
            fh.write('#  bondflux\n')
            fh.write('#  fluxdir Z\n')
            fh.write('#  fluxsurf -5.0 5.0\n')
            fh.write('#  fluxthres 0.3\n')
            fh.write('#  bc_scale_area\n')
            fh.write('#  atomgroup %i-%i\n' % tuple(np.array(self.parser.zmat.electrodes['M'])+1))
            fh.write('#  #read_green\n')
            fh.write('#  #print_green\n')
            fh.write('#$end\n\n')
            fh.write('$subsystem\n')
            fh.write('  print_molden T\n')
            fh.write('  print_diag_central T\n')
            fh.write('  do_diag_central T\n')
            fh.write('  moldeninfile %s.molden.input\n' % os.path.splitext(os.path.basename(self.parser.fn))[0])
            fh.write('$end\n\n')
