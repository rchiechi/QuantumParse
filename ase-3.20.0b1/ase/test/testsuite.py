import os
import sys
from subprocess import Popen
import importlib
from pathlib import Path
import warnings
import argparse
from multiprocessing import cpu_count

from ase.calculators.calculator import names as calc_names
from ase.cli.info import print_info
from ase.cli.main import CLIError

testdir = Path(__file__).parent
datadir = (testdir / 'data').resolve()


def all_test_modules_and_groups():
    names = []
    groups = {}
    for abspath in testdir.rglob('test_*.py'):
        path = abspath.relative_to(testdir)
        name = str(path).rsplit('.', 1)[0].replace('/', '.')
        if str(path.parent) != '.':
            groupname = str(path.parent).replace('/', '.')
            groups.setdefault(groupname, []).append(name)
        else:
            names.append(name)
    return names, groups


def test(calculators=tuple(), jobs=0, verbose=False,
         stream='ignored', strict='ignored'):
    """Run the tests programmatically.

    This is here for compatibility and perhaps convenience."""
    from ase.cli.main import main

    if stream != 'ignored':
        warnings.warn('Ignoring old "stream" keyword', FutureWarning)
    if strict != 'ignored':
        warnings.warn('Ignoring old "strict" keyword', FutureWarning)

    args = ['test']
    if verbose:
        args += ['--verbose']
    if calculators:
        args += ['--calculators={}'.format(','.join(calculators))]
    if jobs:
        args += '--jobs={}'.format(jobs)

    main(args=args)


def have_module(module):
    return importlib.find_loader(module) is not None


MULTIPROCESSING_MAX_WORKERS = 32
MULTIPROCESSING_DISABLED = 0
MULTIPROCESSING_AUTO = -1


def choose_how_many_workers(jobs):
    if jobs == MULTIPROCESSING_AUTO:
        if have_module('xdist'):
            jobs = min(cpu_count(), MULTIPROCESSING_MAX_WORKERS)
        else:
            jobs = MULTIPROCESSING_DISABLED
    return jobs


help_calculators = """\
Calculator testing is currently work in progress.  This
notice applies to the calculators abinit, cp2k, dftb, espresso,
lammpsrun, octopus, and siesta.  The goal of this work is to provide
a configuration in which tests are more reproducible.

Most calculators require datafiles such as pseudopotentials
which are available at

  https://gitlab.com/ase/ase-datafiles

Please install this package using e.g.:

  $ pip install git+https://gitlab.com/ase/ase-datafiles.git

The ASE test suite needs to know the exact binaries for each
of the aforementioned programs.  Currently these must be specified as
a JSON dictionary mapping calculator names to executables, e.g.:

  {"cp2k": "cp2k_shell", "lammps": "lmp", "siesta": "/usr/local/bin/siesta"}

The dictionary must reside in ~/.ase/executables.json or another path
given by the environment variable ASE_EXECUTABLE_CONFIGFILE."""

class CLICommand:
    """Run ASE's test-suite.

    Requires the pytest package.  pytest-xdist is recommended
    in addition as the tests will then run in parallel.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            '-c', '--calculators', default='',
            help='comma-separated list of calculators to test; '
            'see --help-calculators')
        parser.add_argument('--help-calculators', action='store_true',
                            help='show extended help about calculator tests '
                            'and exit')
        parser.add_argument('--list', action='store_true',
                            help='print all tests and exit')
        parser.add_argument('--list-calculators', action='store_true',
                            help='print all calculator names and exit')
        parser.add_argument(
            '-j', '--jobs', type=int, metavar='N',
            default=MULTIPROCESSING_AUTO,
            help='number of worker processes.  If pytest-xdist is available,'
            ' defaults to all available processors up to a maximum of {}.  '
            '0 disables multiprocessing'
            .format(MULTIPROCESSING_MAX_WORKERS))
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='write test outputs to stdout.  '
                            'Mostly useful when inspecting a single test')
        parser.add_argument('--strict', action='store_true',
                            help='convert warnings to errors.  '
                            'This option currently has no effect')
        parser.add_argument('--fast', action='store_true',
                            help='skip slow tests')
        parser.add_argument('--coverage', action='store_true',
                            help='measure code coverage.  '
                            'Requires pytest-cov')
        parser.add_argument('--nogui', action='store_true',
                            help='do not run graphical tests')
        parser.add_argument('tests', nargs='*',
                            help='specify particular test files '
                            'or directories')
        parser.add_argument('--pytest', nargs=argparse.REMAINDER,
                            help='forward all remaining arguments to pytest.  '
                            'See pytest --help')

    @staticmethod
    def run(args):
        print_info()
        print()

        if args.help_calculators:
            print(help_calculators)
            sys.exit(0)

        if args.list_calculators:
            for name in calc_names:
                print(name)
            sys.exit(0)

        if args.nogui:
            os.environ.pop('DISPLAY')

        pytest_args = ['--pyargs', '-v']

        def add_args(*args):
            pytest_args.extend(args)

        if args.list:
            add_args('--collect-only')

        jobs = choose_how_many_workers(args.jobs)
        if jobs:
            add_args('--numprocesses={}'.format(jobs))

        if args.fast:
            add_args('-m', 'not slow')

        if args.coverage:
            add_args('--cov=ase',
                     '--cov-config=.coveragerc',
                     '--cov-report=term',
                     '--cov-report=html')

        if args.tests:
            names, groups = all_test_modules_and_groups()

            testnames = []
            for arg in args.tests:
                if arg in groups:
                    testnames += groups[arg]
                else:
                    testnames.append(arg)

            for testname in testnames:
                add_args('ase.test.{}'.format(testname))
        else:
            add_args('ase.test')

        if args.calculators:
            add_args(f'--calculators={args.calculators}')

        if args.verbose:
            add_args('--capture=no')

        if args.pytest:
            add_args(*args.pytest)

        print()
        calcstring = args.calculators if args.calculators else 'none'
        print('Enabled calculators: {}'.format(calcstring))
        print()
        print('About to run pytest with these parameters:')
        for line in pytest_args:
            print('    ' + line)

        if not have_module('pytest'):
            raise CLIError('Cannot import pytest; please install pytest '
                           'to run tests')

        # We run pytest through Popen rather than pytest.main().
        #
        # This is because some ASE modules were already imported and
        # would interfere with code coverage measurement.
        # (Flush so we don't get our stream mixed with the pytest output)
        sys.stdout.flush()
        proc = Popen([sys.executable, '-m', 'pytest'] + pytest_args,
                     cwd=str(testdir))
        exitcode = proc.wait()
        sys.exit(exitcode)
