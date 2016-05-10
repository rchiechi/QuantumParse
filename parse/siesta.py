from parse import xyz
from util import elements

class Parser(xyz.Parser):

    def _parsezmat(self):

        return

    def _atomlabels(self,fh):
        inblock = False
        fh.seek(0)
        for l in fh.read():
            if '%block chemicalspecieslabel' in l.lower():
                inblock = True
            elif '%endblock chemicalspecieslabel' in l.lower():
                inblock = False
            i,atom,n = l.replace('\t',' ').strip().split(' ')
            #element = elements[n]
        fh.write("%block ChemicalSpeciesLabel\n")
        self.atomnum = {}
        i = 1
        for atom in self.parser.zmat.atoms.unique():
            fh.write("\t%s %s %s\n" % (i, atom, atomicNumber[atom]))
            self.atomnum[atom]=i
            i+=1
        fh.write("%endblock ChemicalSpeciesLabel\n")
