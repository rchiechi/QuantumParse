from parse import xyz
import logging
import pandas as pd
from util import elements


class Parser(xyz.Parser):

    breaks = ('natoms=','stoichiometry')

