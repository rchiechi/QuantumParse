from parse import xyz
import logging
from util import elements


class Parser(xyz.Parser):

    breaks = ('--link1--','natoms=','stoichiometry')

