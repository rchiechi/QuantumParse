from output import xyz
from util import atomicNumber 

class Writer(xyz.Writer):
    ext = '.fdf'
    
    def _section(self,s):
        h ='# ---------------------------------------------------------------------------'
        return '%s\n# %s\n%s\n' % (h,s,h)

    def _writehead(self,fh):
        fh.write(self._section('Name and Label'))
        fh.write("SystemName %s\n" % self.jobname)
        fh.write("SystemLabel %s\n" % self.jobname)
        fh.write(self._section('Species and Atoms'))
        fh.write("NumberOfAtoms %s\n" % len(self.parser.zmat))
        fh.write("NumberOfSpecies %s\n" % len(self.parser.zmat.atoms.unique()))
        fh.write("%block ChemicalSpeciesLabel\n")
        self.atomnum = {}
        i = 1
        for atom in self.parser.zmat.atoms.unique():
            fh.write("\t%s %s %s\n" % (i, atomicNumber[atom], atom) )
            self.atomnum[atom]=i
            i+=1
        fh.write("%endblock ChemicalSpeciesLabel\n")
        fh.write("PAO.EnergyShift         0.010 Ry\n")
        fh.write("%block PAO.BasisSizes\n")
        for atom in self.parser.zmat.atoms.unique():
            if atom in ("Au","Ag"):
                fh.write("\t%s SZ\n" % atom)
            else:
                fh.write("\t%s DZP\n" % atom)
        fh.write("%endblock PAO.BasisSizes\n")
        fh.write(self._section('Lattice'))
        fh.write("#LatticeConstant         1.000 Ang\n")
        #fh.write("#%block LatticeVectors\n")
        #fh.write("#\t1.0    0.0     0.0\n")
        #fh.write("#\t0.0    1.0     0.0\n")
        #fh.write("#\t0.0    0.0     1.0\n")
        #fh.write("#%endblock LatticeVectors\n")
        #fh.write("#AtomicCoordinatesFormat ScaledCartesian\n")
    def _writezmat(self,fh):
        fh.write(self._section('Atomic Coordinates'))
        fh.write("AtomicCoordinatesFormat Ang\n")
        fh.write("%block AtomicCoordinatesAndAtomicSpecies\n")
        i = 1
        for row in self.parser.zmat.iterrows():
            fh.write("\t%.8f\t%.8f\t%.8f\t%s\t%s\t%s\n" % 
                    (row[1].x,row[1].y,row[1].z,self.atomnum[row[1].atoms],row[1].atoms,i))
            i += 1
        fh.write("%endblock AtomicCoordinatesAndAtomicSpecies\n")
    def _writetail(self,fh):
        fh.write(self._section('DFT'))
        fh.write("%block kgrid_Monkhorst_Pack\n\t1    0    0    0.0\n\t0    1    0    0.0\n\t0    0    1    0.0\n%endblock kgrid_Monkhorst_Pack\n")
        fh.write("xc.functional           GGA\n")
        #fh.write("xc.authors              revPBE\n")
        fh.write("xc.authors              BLYP\n")
        fh.write("MeshCutoff              200. Ry\n")
        fh.write("SolutionMethod          diagon\n")
        fh.write("ElectronicTemperature   300 K\n")
        fh.write("MaxSCFIterations        1000\n")
        fh.write("DM.MixingWeight         0.02\n")
        fh.write("DM.NumberPulay          8\n")
        fh.write("DM.MixSCF1              T\n")
        fh.write("DM.Tolerance            1.d-4\n")
        fh.write("WriteMullikenPop        1\n")
        fh.write("WriteWaveFunctions      T\n")
        fh.write("WriteEigenvalues        T\n")
        fh.write("UseSaveData             T\n")
        fh.write("SaveHS                  T\n")
        fh.write("SCFMustConverge         T\n")
        fh.write("MaxSCFIterations      128\n")
        fh.write("%block LocalDensityOfStates\n")
        fh.write("\t-5.00  5.00   eV\n")
        fh.write("%endblock LocalDensityOfStates\n")
        fh.write("%block ProjectedDensityOfStates\n")
        fh.write("\t-20.00  20.00  0.200  1000  eV\n")
        fh.write("%endblock ProjectedDensityOfStates\n")
        if self.opts.transport:
                fh.write(self._section('Gollum'))
                fh.write("Gollum                  EMol\n")
    
