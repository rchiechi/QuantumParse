import logging
import pandas as pd
from util import elements

class Parser:
  
    zmat = pd.DataFrame()
    iorbs = pd.DataFrame()
    logger = logging.getLogger('default')

    def __init__(self,opts,fn):
        self.fn = fn
        self.opts = opts

    def __parsezmat(self):
        zmat = {'atoms':[],'x':[],'y':[],'z':[]}
        with open(self.fn) as fh:
            for l in fh.readlines():
                row = []
                for _l in l.replace('\t',' ').split(' '):
                    if _l.strip(): row.append(_l.strip())
                if not row: continue
                elif row[0] not in elements: continue
                if len(row) >= 4:
                    try:
                        zmat['x'].append(float(row[1]))
                        zmat['y'].append(float(row[2]))
                        zmat['z'].append(float(row[3]))
                        zmat['atoms'].append(str(row[0]))
                    except ValueError:
                        self.logger.warn("Error parsing line in Z-matrix in %s" % self.fn)
        self.__zmattodf(zmat)

    def __zmattodf(self,zmat):
        if self.opts.sortaxis:
            self.logger.debug('Sorting Z-matrix by column %s' % self.opts.sortaxis)
            self.zmat = pd.DataFrame(zmat).sort_values(self.opts.sortaxis)
        else:
            self.zmat = pd.DataFrame(zmat)
        if not len(self.zmat):
            self.logger.error('Empty Z-matrix parsed from %s' % self.fn)
            import sys
            sys.exit()
        self.logger.info('Parsed a Z-matrix with %s atoms.' % len(self.zmat))
    def parseZmatrix(self):
        self.__parsezmat()

    
