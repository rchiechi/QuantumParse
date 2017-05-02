from output import xyz

class Writer(xyz.Writer):

    def _writetail(self,xyzfh):
        bn = xyzfh.name[:-4]
        self.logger.debug('Base filname: %s' % bn)
        fn = bn+'.inp'
        self.logger.info('Writing to %s' % fn)
        mult = xyz.Writer.getMultiplicity(self.parser.zmat)
        with open(fn,'wt') as fh:
            if self.opts.transport:
                fh.write('! DFT B3LYP/G LANL2DZ vdwgrid3 SlowConv\n') 
                fh.write('# ORCA 3\n#! DFT B3LYP/G DUNNING-DZP ECP{LANL2,LANLDZ} vdwgrid3 SlowConv\n') 
                fh.write('%scf MaxIter 1000 end\n') 
            else:
                fh.write('! DFT B3LYP/G Def2-TZVP def2-TZVP\n') 
                fh.write('# ORCA 3\n#! DFT B3LYP/G Def2-TZVP ECP{def2-TZVP}\n') 
            fh.write('#! NRSCF\n') 
            fh.write('#! AHSCF\n') 
            fh.write('#%method SFitInvertType Diag_Q end\n')
            fh.write('* xyzfile 0 %s %s\n'% (mult,xyzfh.name) )
            fh.write('%pal nprocs 48 end\n')
            fh.write('%maxcore 2048\n')
            fh.write('#%plots\n')
            fh.write('#dim1  128\n')
            fh.write('#dim2  128\n')
            fh.write('#dim3  128\n')
            fh.write('#Format Gaussian_Cube\n')
            fh.write('#MO("HOMO.cube",1,0);\n')
            fh.write('#MO("LUMO.cube",2,0);\n')
            fh.write('#end\n')
            if self.opts.transport:
                fh.write('\n$new_job\n')
                fh.write('! DFT B3LYP/G LANL2DZ vdwgrid3 SlowConv\n') 
                fh.write('# ORCA 3\n#! DFT B3LYP/G DUNNING-DZP ECP{LANL2,LANLDZ} vdwgrid3 MOREAD\n') 
                fh.write('%scf MaxIter 1 end\n')
                fh.write('#%method SFitInvertType Diag_Q end\n')
                fh.write('* xyzfile 0 %s %s\n' % (mult,xyzfh.name) )
                fh.write('%%MoInp "%s.gbw"\n' % bn)
                fh.write('%%base "%s_T"\n' % bn)
                fh.write('%output\n')
                fh.write('  Print[P_Iter_F] 1\n')
                fh.write('  Print[P_Overlap] 1\n')
                fh.write('  Print[P_Mos] 1\n')
                fh.write('  Print[P_InputFile] 1\n')
                fh.write('end\n')
            
