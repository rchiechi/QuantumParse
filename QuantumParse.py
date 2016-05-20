#!/usr/bin/env python3.5

import os,sys

try:
    import pip
except ImportError:
    print('You don\'t have pip installed. You will need pip to istall other dependencies.')
    sys.exit(1)

prog = os.path.basename(sys.argv[0]).replace('.py','')
# Need to make this check because ase does not check for dependencies like matplotlib at import
installed = [package.project_name for package in pip.get_installed_distributions()]
required = ['colorama','ase','matplotlib']
for pkg in required:
    if pkg not in installed:
        print('You need to install %s to use %s.' % (pkg,prog))
        print('e.g., sudo -H pip3 install --upgrade %s' % pkg)
        sys.exit(1)

import argparse
import logging
import importlib
from colorama import init,Fore,Back,Style

# Setup colors
init(autoreset=True)

# Parse args
desc = 'Convert between quantum chemistry software file formats.'

# Find parsers and writers
writers,parsers = [],[]
absdir = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(os.path.join(absdir,'output')):
    if f[0] in ('.','_'): continue
    writers.append(f[:-3])
for f in os.listdir(os.path.join(absdir,'parse')):
    if f[0] in ('.','_'): continue
    parsers.append(f[:-3])

parser = argparse.ArgumentParser(description=desc,formatter_class=argparse.ArgumentDefaultsHelpFormatter,add_help=False)

parser.add_argument('infiles', type=str, nargs='*', default=[], 
    help='Datafiles to parse.')
parser.add_argument('-i','--informat', default='guess',
    choices = parsers,
    #choices=('guess','orca','xyz','gaussian','siesta'),
    help="Input file format.")
parser.add_argument('-o','--outformat', required=True,
    choices = writers,
    #choices=('artaios','orca','xyz','gaussian','siesta','gollum'),
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
parser.add_argument('--writeelectrodes', action='store_true', default=False,
    help="Write copies of the electrodes to separate files.")
parser.add_argument('--project', action='store_true', default=False,
    help="Project the molecule along the z-axis.")
parser.add_argument('--build', default=None, choices=('Au','Ag'),
    help="Build electrodes comprising this atom onto the ends of the input molecule.")
parser.add_argument('--size', type=str, default='2,2,4',
    help="Size of electrodes (x,y,z).")


opts=parser.parse_args()

rootlogger = logging.getLogger()
loghandler = logging.StreamHandler()
loghandler.setFormatter(logging.Formatter(\
    fmt=Fore.GREEN+'%(name)s'+Fore.CYAN+' %(levelname)s '+Fore.YELLOW+'%(message)s'+Style.RESET_ALL))
rootlogger.addHandler(loghandler)
rootlogger.setLevel(getattr(logging,opts.loglevel.upper()))
logger = logging.getLogger(prog)

if not len(opts.infiles):
    logger.error("No input files!")
    sys.exit()

try:
    opts.size = tuple(map(int,opts.size.split(',')))
    if len(opts.size) != 3:
        raise ValueError
except ValueError:
    logger.error('%s is not a valid size.' % str(opts.size))
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
    elif ext in ('fdf'):
        opts.informat = 'siesta'
    else:
        logger.error("Could not determine input file format")
        sys.exit()

if opts.transport:
    if opts.outformat == 'siesta':
        logger.debug('Setting electrodes to true for transiesta.')
        opts.writeelectrodes = True

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
