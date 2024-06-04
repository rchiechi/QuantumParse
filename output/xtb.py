from output import xyz

class Writer(xyz.Writer):

    def _writetail(self,fh):
        bn = fh.name[:-4]
        self.logger.debug('Base filname: %s', bn)
        with open(f'{bn}.inp', 'w') as fh:
            fh.write("$constrain\n"
                    f"elements: {self.opts.build}\n"
                     "$end\n"
                     "$write\n"
                     "mos=false\n"
                     "fod=false\n"
                     "orbital energies=true\n"
                     "$end")