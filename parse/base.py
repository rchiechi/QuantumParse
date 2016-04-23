import logging
import pandas as pd


class Parser:
    
    zmat = pd.DataFrame()
    iorbs = pd.DataFrame()
    logger = logging.getLogger('default')

    def __init__(self,fn):
        self.fn = fn

    def __parsezmat(self):
        '''Override me'''
        return

    def GetZmatrix(self):
        self.__parsezmat()
        self.logger.debug(str(self.zmat))
