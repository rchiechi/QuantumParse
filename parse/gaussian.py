import logging
import pandas as pd
from parse import base

class Parser(base.Parser):

    def __parsezmat(self):
        zmat = {'Atoms':[],'X':[],'Y':[],'Z':[]}
        with open(self.fn) as fh:
            for l in fh.readlines():
                row = []
                for _l in l.split(' '):
                    if _l.strip():
                       row.append(_l.strip())
                if len(row) == 5:
                    # Drop -1 colum
                    del(row[1])
                if len(row) == 4:
                    try:
                        zmat['Atoms'].append(str(row[0]))
                        zmat['X'].append(float(row[1]))
                        zmat['Y'].append(float(row[2]))
                        zmat['Z'].append(float(row[3]))
                    except ValueError:
                        self.logger.warn("Error parsing line in Z-matrix in %s" % self.fn)
        self.zmat = pd.DataFrame(zmat)
