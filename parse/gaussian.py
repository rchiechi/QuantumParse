from parse import xyz
import logging
import pandas as pd
from util import elements


class Parser(xyz.Parser):

    def __parsezmat(self):
        zmat = {'atoms':[],'x':[],'y':[],'z':[]}
        with open(self.fn) as fh:
            for l in fh.readlines():
                row = []
                for _l in l.replace('\t',' ').split(' '):
                    if _l.strip():
                        row.append(_l.strip())
                if not row:
                    continue
                elif row[0] == 'NAtoms=':
                    self.logger.debug('Z-matrix should end up with %s atoms.' % row[1])
                    break
                elif row[0] not in elements:
                    continue
                if len(row) == 5:
                    # Drop -1 colum if it exists
                    del(row[1])
                if len(row) == 4:
                    try:
                        zmat['x'].append(float(row[1]))
                        zmat['y'].append(float(row[2]))
                        zmat['z'].append(float(row[3]))
                        zmat['atoms'].append(str(row[0]))
                    except ValueError:
                        if '.log' not in self.fn:
                            self.logger.warn("Error parsing line in Z-matrix in %s" % self.fn)
                            print(row)
        self.__zmattodf(zmat)
       
