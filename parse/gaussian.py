from parse import xyz
import logging
import cclib
import numpy as np

class Parser(xyz.Parser):
    begin = ('Symbolic Z-matrix:','Normal termination of Gaussian')
    breaks = ('--link1--','natoms=','stoichiometry','Stoichiometry','Standard orientation:')

    def __dotransport(self):
        '''Returns overlap, hamiltonian'''
        self.logger.info("Parsing Hamiltonian and Overlap")
        fh = open(self.fn)
        if hasattr(self, 'ccparsed'):
            if not self.ccparsed:
                self.ccparsed = cclib.io.ccread(fh)
        fh.seek(0)
        infile = True
        while infile:
            line = next(fh)
            if line[1:7] == "******" and (line[8:24] == "Core Hamiltonian" or line[11:27] == "Core Hamiltonian" ):
                self.logger.debug(line.strip())
                hamiltonian = np.zeros((self.ccparsed.nbasis, self.ccparsed.nbasis), "d")
                base = 0
                colmNames = next(fh)
                while base < self.ccparsed.nbasis:
                    for i in range(self.ccparsed.nbasis-base):  # Fewer lines this time
                        line = next(fh)
                        parts = line.split()
                        for j in range(len(parts)-1):  # Some lines are longer than others
                            k = float(parts[j+1].replace("D", "E"))
                            hamiltonian[base+j, i+base] = k
                            hamiltonian[i+base, base+j] = k
                    base += 5
                    colmNames = next(fh)
                hamiltonian = np.array(hamiltonian, "d")
                infile = False
        #print(hamiltonian)
        self.fm = hamiltonian
        self.ol = self.ccparsed.aooverlaps
