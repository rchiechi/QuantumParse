#!/usr/bin/env python3
'''
Copyright (C) 2015 Ryan Chiechi <r.c.chiechi@rug.nl>
Description:
        This program parses raw current-voltage data obtained from
        molecular tunneling junctions. It is specifically designed
        with EGaIn in mind, but may be applicable to other systems.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import os,sys
import argparse
import logging
import importlib
from colorama import init,Fore,Back,Style


# Setup colors
init(autoreset=True)

# Parse args
desc = 'Convert between quantum chemistry software file formats.'

parser = argparse.ArgumentParser(description=desc,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infiles', type=str, nargs='*', default=[], 
    help='Datafiles to parse.')
parser.add_argument('-i','--informat', default='guess',
    choices=('guess','orca','xyz','gaussian'),
    help="Input file format.")
parser.add_argument('-o','--outformat', required=True,
    choices=('artaios','orca','xyz','gaussian'),
    help="Output file format.")
parser.add_argument('-l','--loglevel', default='info', 
    choices=('info','warn','error','debug'),
    help="Set the logging level.")


opts=parser.parse_args()

prog = os.path.basename(sys.argv[0]).replace('.py','')
logger = logging.getLogger('default')
loghandler = logging.StreamHandler()
loghandler.setFormatter(logging.Formatter(\
    fmt=Fore.GREEN+prog+Fore.CYAN+' %(levelname)s '+Fore.YELLOW+'%(message)s'+Style.RESET_ALL))
logger.addHandler(loghandler)
logger.setLevel(getattr(logging,opts.loglevel.upper()))

if not len(opts.infiles):
    logger.error("No input files!")
    sys.exit()

if opts.informat == 'guess':
    logger.debug('Guessing input file format')
    ext = opts.infiles[0].split('.')[-1].lower()
    if ext in ('com'):
        opts.informat = 'gaussian'
    elif ext in ('inp'):
        opts.informat = 'orca'
    elif ext in ('xyz'):
        opts.informat = 'xyz'
    else:
        logger.error("Could not determine input file format")
        sys.exit()

logger.debug("Input format: %s, Output format: %s" % (opts.informat,opts.outformat))
parsers = []
for fn in opts.infiles:
    parsers.append(importlib.import_module('parse.%s' % opts.informat).Parser(fn))

if opts.outformat in ('xyz','orca','gaussian'):
    for p in parsers:
        p.GetZmatrix()
