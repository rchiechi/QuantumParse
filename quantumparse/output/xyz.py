import os
import logging
import importlib
from ase.io import write as ase_write


# TODO Files are written in CWD instead of input (at least for siesta)

logger = logging.getLogger(__name__)

class Writer:

    ext = '.xyz'
    

    def __init__(self,parser):
        self.parser = parser
        self.opts = parser.opts
        if self.opts.jobname:
            self.jobname = self.opts.jobname
            logger.debug('Setting jobname to %s' % self.jobname)
        else:
            # self.jobname = ''.join(self.parser.fn.split('.')[0:-1])
            self.jobname = ''.join(os.path.basename(self.parser.fn).split('.')[0:-1])
        self.fn = self.jobname+self.ext
        if self.opts.optimize:
            self.parser.zmat.optimize()

    def writeelectrodes(self):
        if not self.parser.haselectrodes():
            logger.warn('Not writing non-existant electrodes.')
            return
        # TODO Hackish way to avoid recursive loop
        opts = self.opts
        opts.writeelectrodes = False
        for e in ('L','R'):
            if self.opts.build and self.opts.size[2] > 2:
                logger.debug('Removing two layers of atoms for Siesta leads')
                s = Writer.trimElectrodes(self.parser.zmat, e, self.opts.size)
                # if e == 'L':
                #    s = (self.parser.zmat.electrodes[e][0],
                #     self.parser.zmat.electrodes[e][1]-int(self.opts.size[0]*self.opts.size[1]*2))
                # elif e == 'R':
                #    s = (self.parser.zmat.electrodes[e][0]+int(self.opts.size[0]*self.opts.size[1]*2),
                #     self.parser.zmat.electrodes[e][1])
            else:
                s = self.parser.zmat.electrodes[e]
            opts.jobname = 'lead'+e
            parser = importlib.import_module('parse.%s' % opts.informat).Parser(opts,self.fn)
            parser.setZmat(self.parser.getZmat()[s[0]:s[1]+1])
            writer = importlib.import_module('output.%s' % opts.outformat).Writer(parser)
            writer.write()

    @classmethod
    def trimElectrodes(cls, zmat, e, size):
        if e == 'L':
            s = (zmat.electrodes[e][0],
                 zmat.electrodes[e][1] - int(size[0]*size[1]*2))
        elif e == 'R':
            s = (zmat.electrodes[e][0]+int(size[0]*size[1]*2),
                 zmat.electrodes[e][1])
        return s

    def write(self):
        if os.path.exists(self.fn) and not self.opts.overwrite:
            logger.error('Not overwriting %s' % self.fn)
        else:
            logger.info('Writing to: %s' % self.fn)
            with open(self.fn, 'w') as fh:
                self._writehead(fh)
                self._writezmat(fh)
                self._writetail(fh)
            if self.opts.writeelectrodes:
                self.writeelectrodes()
        if self.opts.png:
            logger.info('Writing %s.png' % self.jobname)
            try:
                # pass
                ase_write('%s.png' % self.jobname,self.parser.zmat,rotation='90y')
            except ValueError as msg:
                logger.warn("Error writing png file: %s" % str(msg))
            except TypeError as msg:
                logger.warn("Error writing png file %s" % str(msg))
            except ImportError as msg:
                logger.error("Error importing _png module %s", str(msg))

    def _writehead(self,fh):
        fh.write('%s\n' % len(self.parser.zmat))
        fh.write('%s\n' % self.jobname)

    def _writezmat(self,fh):
        if self.parser.zmat.optimized is not None:
            self.parser.zmat.optimized.write(fh)
        else:
            self.parser.zmat.write(fh)
        # , sep='\t',
        #        header=False, index=False,
        #        float_format='%.8f')

    def _writetail(self,fh):
        return

    @classmethod
    def getMultiplicity(cls,zmat):
        n = 0
        for atom in zmat:
            n += atom.number
        return n % 2 + 1
