#!/usr/bin/env python3

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
