#! /usr/bin/env python3


import cclib
import numpy as np

fh = open('/Users/rchiechi/Desktop/AC_Au.log')
gf = cclib.io.ccread(fh)
fh.seek(0)


infile = True
while infile:
    line = next(fh)
    if line[1:7] == "******" and (line[8:24] == "Core Hamiltonian" or line[11:27] == "Core Hamiltonian" ):
        print(line)
        hamiltonian = np.zeros((gf.nbasis, gf.nbasis), "d")
        base = 0
        colmNames = next(fh)
        while base < gf.nbasis:
            for i in range(gf.nbasis-base):  # Fewer lines this time
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
print(hamiltonian)
