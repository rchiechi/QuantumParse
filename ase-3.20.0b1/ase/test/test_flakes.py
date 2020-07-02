# Run flake8 on main source dir and documentation.
import sys
from collections import defaultdict
from pathlib import Path
from subprocess import Popen, PIPE

import pytest

import ase

pytest.importorskip('flake8')


asepath = Path(ase.__path__[0])  # type: ignore


max_errors = {
    # do not compare types, use 'isinstance()'
    'E721': 0,
    # multiple imports on one line
    'E401': 0,
    # multiple spaces before keyword
    'E272': 0,
    # continuation line under-indented for hanging indent
    'E121': 0,
    # whitespace before '('
    'E211': 0,
    # continuation line with same indent as next logical line
    'E125': 0,
    # comparison to True should be 'if cond is True:' or 'if cond:'
    'E712': 0,
    # 'name' imported but unused
    'F401': 0,
    # no newline at end of file
    'W292': 0,
    # missing whitespace after keyword
    'E275': 0,
    # multiple spaces after operator
    'E222': 0,
    # missing whitespace around modulo operator
    'E228': 0,
    # expected 1 blank line before a nested definition, found 0
    'E306': 0,
    # test for membership should be 'not in'
    'E713': 4,
    # multiple statements on one line (colon)
    'E701': 5,
    # indentation is not a multiple of four (comment)
    'E114': 5,
    # unexpected indentation (comment)
    'E116': 5,
    # comparison to None should be 'if cond is None:'
    'E711': 5,
    # expected 1 blank line, found 0
    'E301': 8,
    # multiple spaces after keyword
    'E271': 8,
    # test for object identity should be 'is not'
    'E714': 8,
    # closing bracket does not match visual indentation
    'E124': 8,
    # too many leading '#' for block comment
    'E266': 10,
    # over-indented
    'E117': 11,
    # indentation contains mixed spaces and tabs
    'E101': 12,
    # indentation contains tabs
    'W191': 13,
    # closing bracket does not match indentation of opening bracket's line
    'E123': 14,
    # multiple spaces before operator
    'E221': 16,
    # whitespace before '}'
    'E202': 19,
    # whitespace after '{'
    'E201': 20,
    # inline comment should start with '# '
    'E262': 20,
    # the backslash is redundant between brackets
    'E502': 30,
    # continuation line missing indentation or outdented
    'E122': 31,
    # indentation is not a multiple of four
    'E111': 36,
    # do not use bare 'except'
    'E722': 38,
    # whitespace before ':'
    'E203': 38,
    # blank line at end of file
    'W391': 49,
    # continuation line over-indented for hanging indent
    'E126': 48,
    # multiple spaces after ','
    'E241': 50,
    # continuation line under-indented for visual indent
    'E128': 54,
    # continuation line over-indented for visual indent
    'E127': 60,
    # missing whitespace around operator
    'E225': 61,
    # ambiguous variable name 'O'
    'E741': 77,
    # too many blank lines (2)
    'E303': 237,
    # expected 2 blank lines after class or function definition, found 1
    'E305': 83,
    # module level import not at top of file
    'E402': 97,
    # at least two spaces before inline comment
    'E261': 97,
    # expected 2 blank lines, found 1
    'E302': 111,
    # unexpected spaces around keyword / parameter equals
    'E251': 117,
    # trailing whitespace
    'W291': 222,
    # block comment should start with '# '
    'E265': 246,
    # missing whitespace after ','
    'E231': 465,
    # missing whitespace around arithmetic operator
    'E226': 563,
    # line too long (93 > 79 characters)
    'E501': 762}


def have_documentation():
    import ase
    ase_path = Path(ase.__path__[0])
    doc_path = ase_path.parent / 'doc/ase/ase.rst'
    return doc_path.is_file()


@pytest.mark.slow
def test_flake8():
    if not have_documentation():
        pytest.skip('ase/doc not present; '
                    'this is probably an installed version ')

    args = [
        sys.executable,
        '-m',
        'flake8',
        str(asepath),
        str((asepath / '../doc').resolve()),
        '--exclude',
        str((asepath / '../doc/build/*').resolve()),
        '--ignore',
        'E129,W293,W503,W504,E741',
        '-j',
        '1'
    ]
    proc = Popen(args, stdout=PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('utf8')

    errors = defaultdict(int)
    files = defaultdict(int)
    offenses = defaultdict(list)
    descriptions = {}
    for stdout_line in stdout.splitlines():
        tokens = stdout_line.split(':', 3)
        filename, lineno, colno, complaint = tokens
        lineno = int(lineno)
        e = complaint.strip().split()[0]
        errors[e] += 1
        descriptions[e] = complaint
        files[filename] += 1
        offenses[e] += [stdout_line]

    errmsg = ''
    for err, nerrs in errors.items():
        nmaxerrs = max_errors.get(err, 0)
        if nerrs <= nmaxerrs:
            continue
        errmsg += 'Too many flakes: {} (max={})\n'.format(nerrs, nmaxerrs)
        errmsg += 'Offenses:\n' + '\n'.join(offenses[err]) + '\n'

    assert errmsg == '', errmsg
