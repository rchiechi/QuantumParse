import os
from output import xyz


class Writer(xyz.Writer):

    ext = ''

    def __init__(self,parser):
        xyz.Writer.__init__(self,parser)
        self.fn = os.path.join(os.path.split(self.fn)[0],'input')


    def _writehead(self,fh):
        
        fh.write('# name: Mode\n')
        fh.write('# type: scalar\n')
        fh.write('4\n')
        fh.write('# name: Bias\n')
        fh.write('# type: matrix\n')
        fh.write('# rows: 2\n')
        fh.write('# columns: 3\n')
        fh.write('-2 2 200\n')
        fh.write('0    0    0\n')
        fh.write('# name: ERange\n')
        fh.write('# type: matrix\n')
        fh.write('# rows: 1\n')
        fh.write('# columns: 3\n')
        fh.write('-2.  2. 400\n')
        fh.write('# name: EF_shift\n')
        fh.write('# type: scalar\n')
        fh.write('0\n')
        fh.write('# name: leadp\n')
        fh.write('# type: matrix\n')
        fh.write('# rows: 2\n')
        fh.write('# columns: 3\n')
        fh.write('1  1  -1\n')
        fh.write('1  1  1\n')
        fh.write('# name: atom\n')
        fh.write('# type: matrix\n')
        fh.write('# rows: %s\n' % str(len(self.parser.zmat)-1) )
        fh.write('# columns: 4\n')
         
    def _writezmat(self,fh):
        m = {"L":"1 1 1","M":"0 0 0","R":"2 1 2"}
        if not self.parser.haselectrodes():
            self.logger.error("Error finding electrodes!")
            fh.write("Error in QuantumParse!")
            return
        e = self.parser.electrodes
        for a in ('L','M','R'):
            self.logger.debug('%s:%s' % (e[a][0],e[a][1]))
            for row in self.parser.zmat[e[a][0]:e[a][1]].iterrows():
                fh.write('%s %s\n' % (row[0]+1,m[a]))
        
    def _writetail(self,fh):
        self.logger.info("Wrote gollum input to '%s'" %fh.name)

