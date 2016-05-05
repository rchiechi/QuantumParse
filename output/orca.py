from output import xyz

class Writer(xyz.Writer):

    def _writetail(self,xyzfh):
        bn = xyzfh.name[:-4]
        self.logger.debug('Base filname: %s' % bn)
        fn = bn+'.inp'
        self.logger.info('Writing to %s' % fn)
        mult = xyz.Writer.getMultiplicity(self.parser.zmat)
        with open(fn,'wt') as fh:
            fh.write('! DFT B3LYP Def2-TZVP ECP{def2-TZVP}\n') 
            fh.write('* xyzfile 0 %s %s\n'% (mult,xyzfh.name) )
            fh.write('%pal nprocs 48 end\n')
            if not self.opts.transport:
               return
            fh.write('\n$new_job\n')
            fh.write('! DFT B3LYP Def2-TZVP ECP{def2-TZVP} MOREAD\n')
            fh.write('* xyzfile 0 %s %s\n' % (mult,xyzfh.name) )
            fh.write('%%MoInp "%s.gbw"\n' % bn)
            fh.write('%%base "%s_T"\n' % bn)
            fh.write('%output\n')
            fh.write('  Print[P_Iter_F] 1\n')
            fh.write('  Print[P_Overlap] 1\n')
            fh.write('  Print[P_Mos] 1\n')
            fh.write('  Print[P_InputFile] 1\n')
            fh.write('end\n')


