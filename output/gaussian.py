from output import xyz

class Writer(xyz.Writer):
    
    ext = '.com'

    def __writehead(self,fh):
        fh.write('%%chk=%s.chk\n' % self.jobname)
        fh.write('%%nprocshared=%s\n' % self.opts.ncpus)
        fh.write('%%mem=%sGB\n' % self.opts.ncpus)
        fh.write('# b3lyp/lanl2dz GFPrint\n\n')
        fh.write('%s\n\n' % self.jobname)
        fh.write(' 0 %s\n' % xyz.Writer.getMultiplicity(self.parser.zmat) )

    def __writetail(self,fh):
        fh.write('\n')
        if self.opts.transport:
            fh.write('--Link1--\n')
            fh.write('%%chk=%s.chk\n' % self.jobname)
            fh.write('%%nprocshared=%s\n' % self.opts.ncpus)
            fh.write('%%mem=%sGB\n' % self.opts.ncpus)
            fh.write('# b3lyp/lanl2dz guess=read\n')
            fh.write('# iop(5/33=3)\n')
            fh.write('# iop(3/33=1)\n\n')
            fh.write('%s\n\n' % self.jobname)
            fh.write(' 0 %s\n' % xyz.Writer.getMultiplicity(self.parser.zmat) )
            self.__writezmat(fh)
            fh.write('\n')

