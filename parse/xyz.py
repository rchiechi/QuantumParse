import logging
import pandas as pd
from util import elements
from time import sleep
import re

class Parser:
  
    zmat = pd.DataFrame()
    iorbs = pd.DataFrame()
    logger = logging.getLogger('default')
    #When parsing large files set lowercase strings
    #that terminate the z-matrix to avoid looping
    #over the entire file
    breaks = ()
    ws = re.compile('\s+')

    def __init__(self,opts,fn):
        self.fn = fn
        self.opts = opts

    def _parsezmat(self):
        zmat = {'atoms':[],'x':[],'y':[],'z':[]}
        with open(self.fn) as fh:
            for l in fh:
                row = []
                #for _l in l.replace('\t',' ').split(' '):
                for _l in re.split(self.ws,l):
                    if _l.strip(): row.append(_l.strip())
                if not row: continue
                elif row[0].lower() in self.breaks: 
                    self.logger.debug("Hit break in Z-matrix (%s)" % l.strip())
                    break
                elif row[0] not in elements:continue
                if len(row) >= 4:
                    try:
                        x,y,z = map(float,row[1:4])
                        zmat['x'].append(x)
                        zmat['y'].append(y)
                        zmat['z'].append(z)
                        zmat['atoms'].append(str(row[0]))
                    except ValueError:
                        self.logger.debug("Error parsing line in Z-matrix in %s" % self.fn)
                        self.logger.debug(' '.join(row))
        self._zmattodf(zmat)

    def _zmattodf(self,zmat):
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
        self._parsezmat()

    
