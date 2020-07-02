#!/usr/bin/env python3
"""Bash completion for ase.

Put this in your .bashrc::

    complete -o default -C /path/to/ase/cli/complete.py ase

or run::

    $ ase completion

"""

import os
import sys
from glob import glob


def match(word, *suffixes):
    return [w for w in glob(word + '*')
            if any(w.endswith(suffix) for suffix in suffixes)]


# Beginning of computer generated data:
commands = {
    'band-structure':
        ['-q', '--quiet', '-k', '--path', '-n', '--points', '-o',
         '--output', '-r', '--range'],
    'build':
        ['-M', '--magnetic-moment', '--modify', '-V', '--vacuum', '-v',
         '--vacuum0', '--unit-cell', '--bond-length', '-x',
         '--crystal-structure', '-a', '--lattice-constant',
         '--orthorhombic', '--cubic', '-r', '--repeat', '-g',
         '--gui', '--periodic'],
    'completion':
        [],
    'convert':
        ['-v', '--verbose', '-i', '--input-format', '-o',
         '--output-format', '-f', '--force', '-n',
         '--image-number', '-e', '--exec-code', '-E',
         '--exec-file', '-a', '--arrays', '-I', '--info', '-s',
         '--split-output', '--read-args', '--write-args'],
    'db':
        ['-v', '--verbose', '-q', '--quiet', '-n', '--count', '-l',
         '--long', '-i', '--insert-into', '-a',
         '--add-from-file', '-k', '--add-key-value-pairs', '-L',
         '--limit', '--offset', '--delete', '--delete-keys',
         '-y', '--yes', '--explain', '-c', '--columns', '-s',
         '--sort', '--cut', '-p', '--plot', '--csv', '-w',
         '--open-web-browser', '--no-lock-file', '--analyse',
         '-j', '--json', '-m', '--show-metadata',
         '--set-metadata', '-M', '--metadata-from-python-script',
         '--unique', '--strip-data', '--show-keys',
         '--show-values'],
    'diff':
        ['-r', '--rank-order', '-c', '--calculator-outputs',
         '--max-lines', '-t', '--template', '--template-help',
         '-s', '--summary-functions', '--log-file', '--as-csv'],
    'dimensionality':
        ['--display-all', '--no-merge'],
    'eos':
        ['-p', '--plot', '-t', '--type'],
    'find':
        ['-v', '--verbose', '-l', '--long', '-i', '--include', '-x',
         '--exclude'],
    'gui':
        ['-n', '--image-number', '-r', '--repeat', '-R', '--rotations',
         '-o', '--output', '-g', '--graph', '-t', '--terminal',
         '--interpolate', '-b', '--bonds', '-s', '--scale'],
    'info':
        ['-v', '--verbose', '--formats', '--calculators'],
    'nebplot':
        ['--nimages', '--share-x', '--share-y'],
    'nomad-get':
        [],
    'nomad-upload':
        ['-t', '--token', '-n', '--no-save-token', '-0', '--dry-run'],
    'reciprocal':
        ['-v', '--verbose', '-p', '--path', '-d', '--dimension',
         '--no-vectors', '-k', '--k-points', '-i',
         '--ibz-k-points'],
    'run':
        ['-p', '--parameters', '-t', '--tag', '--properties', '-f',
         '--maximum-force', '--constrain-tags', '-s',
         '--maximum-stress', '-E', '--equation-of-state',
         '--eos-type', '-o', '--output', '--modify', '--after'],
    'test':
        ['-c', '--calculators', '--help-calculators', '--list',
         '--list-calculators', '-j', '--jobs', '-v', '--verbose',
         '--strict', '--fast', '--coverage', '--nogui',
         '--pytest'],
    'ulm':
        ['-n', '--index', '-d', '--delete', '-v', '--verbose']}
# End of computer generated data


def complete(word, previous, line, point):
    for w in line[:point - len(word)].strip().split()[1:]:
        if w[0].isalpha():
            if w in commands:
                command = w
                break
    else:
        if word[:1] == '-':
            return ['-h', '--help', '--version']
        return list(commands.keys()) + ['-h', '--help', '--verbose']

    if word[:1] == '-':
        return commands[command]

    words = []

    if command == 'db':
        if previous == 'db':
            words = match(word, '.db', '.json')

    elif command == 'run':
        if previous == 'run':
            from ase.calculators.calculator import names as words

    elif command == 'build':
        if previous in ['-x', '--crystal-structure']:
            words = ['sc', 'fcc', 'bcc', 'hcp', 'diamond', 'zincblende',
                     'rocksalt', 'cesiumchloride', 'fluorite', 'wurtzite']

    elif command == 'test':
        if previous in ['-c', '--calculators']:
            from ase.calculators.calculator import names as words
        elif not word.startswith('-'):
            from ase.test.testsuite import all_test_modules_and_groups
            names, groups = all_test_modules_and_groups()
            group_completions = [group + '.' for group in groups]
            for group in group_completions:
                if word.startswith(group):
                    return groups[group[:-1]]
            words = names + list(groups) + group_completions

    return words


if sys.version_info[0] == 2:
    import warnings
    warnings.warn('Command-line completion running with python2.  '
                  'Your ASE autocompletion setup is probably outdated.  '
                  'Please consider rerunning \'ase completion\'.')


def main():
    word, previous = sys.argv[2:]
    line = os.environ['COMP_LINE']
    point = int(os.environ['COMP_POINT'])
    words = complete(word, previous, line, point)
    for w in words:
        if w.startswith(word):
            print(w)


if __name__ == '__main__':
    main()
