import os
import pandas as pd
import logging
import importlib
from ase.io import write as ase_write

#TODO Files are written in CWD instead of input (at least for siesta)

class Writer:
    
    ext = '.xyz'
    logger = logging.getLogger('Writer')

    def __init__(self,parser):
        self.parser = parser
        self.opts = parser.opts
        if self.opts.jobname:
            self.jobname = self.opts.jobname
            self.logger.debug('Setting jobname to %s' % self.jobname)
        else:
            #self.jobname = ''.join(self.parser.fn.split('.')[0:-1])
            self.jobname = ''.join(os.path.basename(self.parser.fn).split('.')[0:-1])
        self.fn = self.jobname+self.ext

    def writeelectrodes(self):
        if not self.parser.haselectrodes():
            self.logger.warn('Not writing non-existant electrodes.')
            return
        #TODO Hackish way to avoid recursive loop
        opts = self.opts
        opts.writeelectrodes = False
        for e in ('L','R'):
            if self.opts.build and self.opts.size[2] > 2:
                self.logger.debug('Removing two layers of atoms for Siesta leads')
                if e == 'L':
                    s = (self.parser.zmat.electrodes[e][0],
                     self.parser.zmat.electrodes[e][1]-int(self.opts.size[0]*self.opts.size[1]*2))
                elif e == 'R':
                    s = (self.parser.zmat.electrodes[e][0]+int(self.opts.size[0]*self.opts.size[1]*2),
                     self.parser.zmat.electrodes[e][1])
            else:
                s = self.parser.zmat.electrodes[e]
            opts.jobname = 'lead'+e
            parser = importlib.import_module('parse.%s' % opts.informat).Parser(opts,self.fn)
            parser.setZmat(self.parser.getZmat()[s[0]:s[1]+1])
            writer = importlib.import_module('output.%s' % opts.outformat).Writer(parser)
            writer.write()

    def write(self):
        if os.path.exists(self.fn) and not self.opts.overwrite:
            self.logger.error('Not overwriting %s' % self.fn)
        else:
            self.logger.info('Writing to: %s' % self.fn)
            with open(self.fn, 'w') as fh:
                self._writehead(fh)
                self._writezmat(fh)
                self._writetail(fh)
            if self.opts.writeelectrodes:
                self.writeelectrodes()
        self.logger.info('Writing %s.png' % self.jobname)
        ase_write('%s.png' % self.jobname,self.parser.zmat,rotation='90y')

    def _writehead(self,fh):
        fh.write('%s\n' % len(self.parser.zmat))
        fh.write('%s\n' % self.jobname)

    def _writezmat(self,fh):
        self.parser.zmat.write(fh)
        #, sep='\t', 
        #        header=False, index=False,
        #        float_format='%.8f')

    def _writetail(self,fh):
        return

    @classmethod
    def getMultiplicity(cls,zmat):
        n = 0
        vc = zmat.getAtomCounts()
        for atom in vc:
            if atom.upper() == 'H':
                n += vc[atom]
            else:
                n += vc[atom]*2
        return n%2 + 1
