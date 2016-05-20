from output import xyz
from ase import Atom
from util import *

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
        fh.write("NumberOfSpecies %s\n" % len(self.parser.zmat.unique()))
        fh.write("%block ChemicalSpeciesLabel\n")
        self.atomnum = {}
        i = 1
        for atom in self.parser.zmat.unique():
            fh.write("\t%s %s %s\n" % (i, Atom(atom).number, atom) )
            self.atomnum[atom]=i
            i+=1
        fh.write("%endblock ChemicalSpeciesLabel\n")
        fh.write("PAO.EnergyShift         0.0010 Ry\n")
        fh.write("%block PAO.BasisSizes\n")
        for atom in self.parser.zmat.unique():
            if atom in EATOMS:
                fh.write("\t%s SZ\n" % atom)
            else:
                fh.write("\t%s DZP\n" % atom)
        fh.write("%endblock PAO.BasisSizes\n")
        if self.parser.hasLattice():
            lattice = self.parser.getLattice()
            fh.write(self._section('Lattice'))
            fh.write('%s\n' % lattice['constant'])
            fh.write("%block LatticeVectors\n")
            for v in lattice['vectors']:
                # MUST be FIFO
                fh.write('  %s\n' % v)
            fh.write("%endblock LatticeVectors\n")
        fh.write(self._section('K-grid'))
        if 'lead' in self.opts.jobname:
            zgrid = '60'
        else:
            zgrid = '01'
        fh.write('%block kgrid_Monkhorst_Pack\n')
        fh.write(' 10    0    0    0.0\n')
        fh.write(' 0    10    0    0.0\n')
        fh.write(' 0    0    %s   0.0\n' % zgrid)
        fh.write('%endblock kgrid_Monkhorst_Pack\n')
        #fh.write("#AtomicCoordinatesFormat ScaledCartesian\n")
    def _writezmat(self,fh):
        fh.write(self._section('Atomic Coordinates'))
        fh.write("AtomicCoordinatesFormat Ang\n")
        fh.write("%block AtomicCoordinatesAndAtomicSpecies\n")
        i = 1
        for _a in self.parser.zmat:
            fh.write("\t%.8f\t%.8f\t%.8f\t%s\t%s\t%s\n" % 
                    (_a.x,_a.y,_a.z,self.atomnum[_a.symbol],_a.symbol,i))
            i += 1
        fh.write("%endblock AtomicCoordinatesAndAtomicSpecies\n")
    def _writetail(self,fh):
        fh.write(self._section('DFT'))
        fh.write("xc.functional           GGA\n")
        #fh.write("xc.authors              revPBE\n")
        fh.write("xc.authors              BLYP\n")
        fh.write("MeshCutoff              200. Ry\n")
        fh.write("ElectronicTemperature   300 K\n")
        fh.write("DM.MixingWeight         0.02\n")
        fh.write("DM.NumberPulay          8\n")
        fh.write("DM.MixSCF1              T\n")
        fh.write("DM.Tolerance            1.d-4\n")
        fh.write("WriteMullikenPop        1\n")
        fh.write("WriteWaveFunctions      T\n")
        fh.write("WriteEigenvalues        T\n")
        fh.write("UseSaveData             T\n")
        fh.write("SaveHS                  T\n")
        fh.write("MixHamiltonian          T\n")
        fh.write("DM.UseSaveDM            T\n")
        fh.write("Diag.ParallelOverK    yes\n")
        fh.write("SCFMustConverge         T\n")
        fh.write("MaxSCFIterations      128\n")
        solmeth = 'diagon'
        golmeth = 'Lead'
        if self.opts.transport:
            for a in EATOMS:
                if len(self.parser.zmat.unique()) == 1:
                    self.logger.debug('This looks like an electrode file, not setting up transiesta')
                elif a in self.parser.zmat.get_chemical_symbols():
                    self.logger.debug('This looks like a scattering matrix, setting up transiesta')
                    solmeth = 'transiesta'
                    golmeth = 'EMol'
        fh.write("SolutionMethod      %s\n" % solmeth)
        if solmeth == 'diagon':
            fh.write(self._section('DOS Output'))
            fh.write("%block LocalDensityOfStates\n")
            fh.write("\t-5.00  5.00   eV\n")
            fh.write("%endblock LocalDensityOfStates\n")
            fh.write("%block ProjectedDensityOfStates\n")
            fh.write("\t-20.00  20.00  0.200  1000  eV\n")
            fh.write("%endblock ProjectedDensityOfStates\n")
        if self.opts.transport:
            fh.write(self._section('Gollum'))
            fh.write("#Gollum                  %s\n" % golmeth)
            if solmeth == 'diagon':
                return
            le,re,se = 'leadL.TSHS','leadR.TSHS',self.opts.jobname+'.TSHS'
            fh.write(self._section("Transiesta"))
            fh.write('# GF OPTIONS\n')
            fh.write('TS.ComplexContour.Emin    -30.0 eV\n')
            fh.write('TS.ComplexContour.NPoles       03\n')
            fh.write('TS.ComplexContour.NCircle      30\n')
            fh.write('TS.ComplexContour.NLine        10\n')
            fh.write('# BIAS OPTIONS\n')
            fh.write('TS.biasContour.NumPoints       00\n\n')
            fh.write('# TS OPTIONS\n')
            fh.write('TS.Voltage 0.000000 eV\n\n')
            fh.write('# TBT OPTIONS\n')
            fh.write('%block TBT_Monkhorst_Pack\n')
            fh.write(' 3    0    0    0.0\n')
            fh.write(' 0    3    0    0.0\n')
            fh.write(' 0    0    60   0.5\n')
            fh.write('%endblock TBT_Monkhorst_Pack\n')
            fh.write('TS.TBT.Emin -2.0 eV\n')
            fh.write('TS.TBT.Emax +2.0 eV\n')
            fh.write('TS.TBT.NPoints 500\n')
            fh.write('TS.TBT.NEigen 3\n')
            fh.write('TS.TBT.Eta        0.000001 Ry\n')
            fh.write('TS.TBT.ReUseGF    T\n\n')
            fh.write('# Write hamiltonian\n')
            fh.write('TS.SaveHS   .true.\n\n')
            fh.write('# LEFT ELECTRODE\n')
            fh.write('TS.HSFileLeft  %s\n' % le)
            fh.write('#TS.ReplicateA1Left    1\n')
            fh.write('#TS.ReplicateA2Left    1\n')
            fh.write('#TS.NumUsedAtomsLeft   03\n')
            fh.write('#TS.BufferAtomsLeft    0\n\n')
            fh.write('# RIGHT ELECTRODE\n')
            fh.write('TS.HSFileRight  %s\n' % re)
            fh.write('#TS.ReplicateA1Right   1\n')
            fh.write('#TS.ReplicateA2Right   1\n')
            fh.write('#TS.NumUsedAtomsRight  03\n')
            fh.write('#TS.BufferAtomsRight   0\n\n')
            fh.write('# SCATTERING REGION\n')
            fh.write('#TS.TBT.HSFile    %s\n' % se)
            fh.write('TS.TBT.AtomPDOS       T\n')
