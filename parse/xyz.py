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
    electrodes = {'L':(-1,-1),'M':(-1,-1),'R':(-1,-1),'atom':None}

    def __init__(self,opts,fn):
        self.fn = fn
        self.opts = opts

    def __haveelectrodes(self):
        for a in self.electrodes:
            if -1 in self.electrodes[a] or None in self.electrodes[a]:
                return False
        return True


    def _guesseletrodes(self):
        '''Try to guess electrodes for transport.in.
           only works if molecule crosses 0 along Z.'''
        e1,mol,e2,atom = (-1,-1),(-1,-1),(-1,-1),None
        self.logger.warn('Assuming atoms are sorted along Z.')
        for atom in ('Au','Ag','S'):
            if atom not in self.parser.zmat.atoms.get_values():
                self.logger.debug('No %s electrodes.' % atom)
                continue
            else:
                self.logger.info('Guessing %s electrodes.' % atom)
            molg = self.parser.zmat.atoms[self.parser.zmat.atoms != atom].index
            eg1 = self.parser.zmat.atoms[:molg[0]].index
            eg2 = self.parser.zmat.atoms[molg[-1]+1:].index
            if len(eg1) and len(eg2) and len(molg):
                e1,mol,e2 = eg1,molg,eg2
                break
        #return (e1[0]+1,e1[-1]+1),(mol[0]+1,mol[-1]+1),(e2[0]+1,e2[-1]+1),atom
        self.electrodes['L'] = (e1[0]+1,e1[-1]+1)
        self.electrodes['M'] = (mol[0]+1,mol[-1]+1)
        self.electrodes['R'] = (e2[0]+1,e2[-1]+1)
        self.electrodes['L'] = atom

    def _parsezmat(self):
        zmat = {'atoms':[],'x':[],'y':[],'z':[]}
        with open(self.fn) as fh:
            for l in fh:
                row = []
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
        self._guesselectrodes()
    
