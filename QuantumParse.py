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
parser.add_argument('--overwrite', action='store_true', default=False,
    help="Overwrite output files without asking.")
parser.add_argument('-s','--sortaxis', default=None,
    choices=('x','y','z'),
    help="Sort output zmatrix by given axis.")
parser.add_argument('-l','--loglevel', default='info', 
    choices=('info','warn','error','debug'),
    help="Set the logging level.")
parser.add_argument('-T', '--transport', action='store_true', default=False,
    help="Format orca/gaussian output for transport calculations.")
parser.add_argument('-c', '--ncpus', type=int, default=24,
    help="Number of parallel cpus in output.")
parser.add_argument('--jobname', type=str, default='',
    help="Specify a jobname (and output file name) instead of taking it from the input file name.")


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
    if ext in ('com','log'):
        opts.informat = 'gaussian'
    elif ext in ('inp','out'):
        opts.informat = 'orca'
    elif ext in ('xyz'):
        opts.informat = 'xyz'
    else:
        logger.error("Could not determine input file format")
        sys.exit()

logger.debug("Input format: %s, Output format: %s" % (opts.informat,opts.outformat))
if opts.informat == opts.outformat and not opts.jobname:
    logger.error("You need to set a jobname if input and output formats are the same.")
    sys.exit()
if opts.informat not in ('gaussian','orca') and opts.outformat == 'artaios':
    logger.error("Only gaussian and orca inputs can generate artaios outputs.")
    sys.exit()
if opts.outformat in ('artaios') and opts.sortaxis:
    logger.warn('Sorting the z-matrix and outputting to artaios is a bad idea.')

#elif opts.informat == 'gaussian' and opts.outformat == 'artaios':
#    import subprocess
#    if subprocess.run(['which', 'g09_2unform']).returncode != 0:
#        logger.error("g09_2unform needs to be in your PATH to convert gaussian outputs to artaios inputs.")
#        sys.exit()
#    completed = []
#    for log in opts.infiles:
#        completed.append(subprocess.run(['g09_2unform',log,'1']))
#    sys.exit()

parsers = []
for fn in opts.infiles:
    parsers.append(importlib.import_module('parse.%s' % opts.informat).Parser(opts,fn))

#if opts.outformat in ('xyz','orca','gaussian'):
for p in parsers:
    p.parseZmatrix()
for p in parsers:
    output = (importlib.import_module('output.%s' % opts.outformat).Writer(p))
    output.write()
