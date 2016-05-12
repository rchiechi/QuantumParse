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
    electrodes = {'L':(0,0),'M':(0,0),'R':(0,0),'atom':None}

    def __init__(self,opts,fn):
        self.fn = fn
        self.opts = opts

    def haselectrodes(self):
        for a in ('L','M','R'):
            if self.electrodes[a] == (0,0) or -1 in self.electrodes[a]:
                return False
        return True

    def setZmat(self,zmat):
        self.zmat = zmat

    def getZmat(self):
        return self.zmat

    def _guesselectrodes(self):
        '''Try to guess electrodes for transport.in.
           only works if molecule is sorted along Z.'''
        e1,mol,e2,atom = (-1,-1),(-1,-1),(-1,-1),None
        if not self.opts.sortaxis == 'z':
            self.logger.debug('Guessing electrodes along Z-axis but sort axis is %s!' % self.opts.sortaxis)
        if self.zmat.atoms.head(1).values[0] in ('Au','Ag') and self.zmat.atoms.tail(1).values[0] in ('S'):
            self.logger.warn("You may have an S-atom appended to the end of the Z-matrix instead of a metal.")
        for atom in ('Au','Ag','S'):
            if len(self.zmat[self.zmat.atoms == atom]) == len(self.zmat.atoms):
                self.logger.debug('This looks like an electrode, not guess electrode')
                return
            if atom not in self.zmat.atoms.get_values():
                self.logger.debug('No %s electrodes.' % atom)
                continue
            else:
                self.logger.info('Guessing %s electrodes.' % atom)
            molg = self.zmat.atoms[self.zmat.atoms != atom].index
            eg1 = self.zmat.atoms[:molg[0]].index
            eg2 = self.zmat.atoms[molg[-1]+1:].index
            if len(eg1) and len(eg2) and len(molg):
                self.logger.debug('Parsed %s electrodes.' % atom)
                e1,mol,e2 = eg1,molg,eg2
                break
            else:
                self.logger.debug('Did not parse %s electrode.' % atom)
        self.electrodes['L'] = (e1[0],e1[-1])
        self.electrodes['M'] = (mol[0],mol[-1])
        self.electrodes['R'] = (e2[0],e2[-1])
        self.electrodes['atom'] = atom
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
            idx = []
            for i in range(0,len(zmat['atoms'])):
                idx.append(i)
            self.logger.debug('Sorting Z-matrix by column %s' % self.opts.sortaxis)
            zmat = pd.DataFrame(zmat)
            self.zmat = zmat.sort_values(self.opts.sortaxis)
            self.zmat.index = idx
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
    
