import os
from output import xyz

def get_angular_momentum(elem):
    if elem in ('H', 'He', 'Li', 'Be'):
        return('s')
    if elem in ('B', 'C', 'N', 'O', 'F'):
        return('p')
    else:
        return('d')

class Writer(xyz.Writer):
    'Basic dftb_in.hsd file to do a geometry optimization'

    def _writetail(self, fh):
        fn = 'dftb_in.hsd'
        self.logger.info('Writing to %s', fn)
        with open(fn,'wt') as hsd:
            hsd.write('Geometry = xyzFormat {\n')
            hsd.write(f'<<< {fh.name}.xyz\n')
            hsd.write('}\n')
            hsd.write('Driver = ConjugateGradient {\n')
            hsd.write(f'  MaxSteps = 10000\n  MovedAtoms = 1:-1\n  OutputPrefix = "{fh.name}_opt"\n')
            hsd.write('  #MaxForceComponent = 1.0e-4      # Stop if maximal force below 1.0e-4\n')
            hsd.write('Hamiltonian = DFTB {\n')
            hsd.write('  #ReadInitialCharges = Yes\n  SCCTolerance = 1E-7\n')
            hsd.write('  Scc = Yes\n  MaxSccIterations = 100\n')
            hsd.write('  SlaterKosterFiles = Type2FileNames { # File names with two atom type names\n')
            hsd.write(f'    Prefix = "{os.path.expanduser("~")}/.dftbplus/slakos/3ob-3-1/"    # Prefix before first type name\n')
            hsd.write('    Separator = "-" # Dash between type names\n')
            hsd.write('    Suffix = ".skf" # Suffix after second type name\n')
            hsd.write('  }\n')
            hsd.write('  MaxAngularMomentum {\n')
            for _atom in self.parser.zmat.unique():
                hsd.write(f'    {_atom} = "{get_angular_momentum(_atom)}"\n')
            hsd.write('  }\n}\n')
            hsd.write('Options {\n  WriteDetailedXml = Yes\n}\n')
            hsd.write('Analysis {\n  WriteEigenvectors = Yes\n}\n')
            hsd.write('ParserOptions {\n  ParserVersion=9\n}\n')
