from output import xyz

class Writer(xyz.Writer):

    def _writetail(self,fh):
        bn = fh.name[:-4]
        self.logger.debug('Base filname: %s', bn)
        inpfn = bn+'.inp'
        self.logger.info('Writing to %s', inpfn)
        mult = xyz.Writer.getMultiplicity(self.parser.zmat)
        with open(inpfn,'wt') as inp:
            if self.opts.transport:
                inp.write('! DFT B3LYP/G Def2-SVP SlowConv TightSCF\n')
                # fh.write('# # # ORCA 3\n#! DFT B3LYP/G DUNNING-DZP ECP{LANL2,LANLDZ} vdwgrid3 SlowConv\n# # #\n')
                inp.write('%scf MaxIter 1000 end\n')
            else:
                inp.write('! DFT B3LYP/G Def2-TZVP\n')
                # fh.write('# # # ORCA 3\n#! DFT B3LYP/G Def2-TZVP ECP{def2-TZVP}\n# # #\n')
            inp.write('#%method SFitInvertType Diag_Q end\n')
            inp.write('* xyzfile 0 %s %s\n' % (mult,fh.name))
            inp.write('#%%base "%s"\n' % fh.name.replace('.xyz', '_E'))
            inp.write('%pal nprocs 24 end\n')
            inp.write('%%maxcore %s\n' % self.opts.memory)
            inp.write('#%plots\n')
            inp.write('#dim1  128\n')
            inp.write('#dim2  128\n')
            inp.write('#dim3  128\n')
            inp.write('#Format Gaussian_Cube\n')
            inp.write('#MO("HOMO.cube",1,0);\n')
            inp.write('#MO("LUMO.cube",2,0);\n')
            inp.write('#end\n')
            if self.opts.transport:
                inp.write('\n$new_job\n')
                inp.write('! DFT B3LYP/G Def2-SVP SlowConv TightSCF MOREAD\n')
                # fh.write('# # # ORCA 3\n#! DFT B3LYP/G DUNNING-DZP ECP{LANL2,LANLDZ} vdwgrid3 MOREAD\n# # #\n')
                inp.write('%scf MaxIter 10 end\n')
                inp.write('#%method SFitInvertType Diag_Q end\n')
                inp.write('* xyzfile 0 %s %s\n' % (mult,fh.name))
                inp.write('%%MoInp "%s.gbw"\n' % bn)
                inp.write('%%base "%s_T"\n' % bn)
                inp.write('%output\n')
                inp.write('  Print[P_Iter_F] 1\n')
                inp.write('  Print[P_Overlap] 1\n')
                inp.write('  Print[P_Mos] 1\n')
                inp.write('  Print[P_InputFile] 1\n')
                inp.write('end\n')

