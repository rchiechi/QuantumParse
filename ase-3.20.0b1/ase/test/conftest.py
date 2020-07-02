from pathlib import Path
from subprocess import Popen, PIPE, check_output

import pytest

from ase.utils import workdir
from ase.test.factories import (Factories, CalculatorInputs,
                                make_factory_fixture, get_testing_executables)
from ase.calculators.calculator import (names as calculator_names,
                                        get_calculator_class)


@pytest.fixture(scope='session')
def enabled_calculators(pytestconfig):
    all_names = set(calculator_names)
    opt = pytestconfig.getoption('calculators')

    names = set(always_enabled_calculators)
    if opt:
        for name in opt.split(','):
            if name not in all_names:
                raise ValueError(f'No such calculator: {name}')
            names.add(name)
    return sorted(names)


class Calculators:
    def __init__(self, names):
        self.enabled_names = set(names)

    def require(self, name):
        assert name in calculator_names
        if name not in self.enabled_names:
            pytest.skip(f'use --calculators={name} to enable')


@pytest.fixture(scope='session')
def calculators(enabled_calculators):
    return Calculators(enabled_calculators)


@pytest.fixture(scope='session')
def require_vasp(calculators):
    calculators.require('vasp')


def disable_calculators(names):
    import pytest
    for name in names:
        if name in always_enabled_calculators:
            continue
        try:
            cls = get_calculator_class(name)
        except ImportError:
            pass
        else:
            def get_mock_init(name):
                def mock_init(obj, *args, **kwargs):
                    pytest.skip(f'use --calculators={name} to enable')
                return mock_init

            def mock_del(obj):
                pass
            cls.__init__ = get_mock_init(name)
            cls.__del__ = mock_del



# asap is special, being the only calculator that may not be installed.
# But we want that for performance in some tests.
always_enabled_calculators = set(
    ['asap', 'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p']
)


@pytest.fixture(scope='session', autouse=True)
def monkeypatch_disabled_calculators(request, enabled_calculators):
    from ase.calculators.calculator import names as calculator_names
    test_calculator_names = list(always_enabled_calculators)
    test_calculator_names += enabled_calculators
    disable_calculators([name for name in calculator_names
                         if name not in enabled_calculators])


# Backport of tmp_path fixture from pytest 3.9.
# We want to be compatible with pytest 3.3.2 and pytest-xdist 1.22.1.
# These are provided with Ubuntu 18.04.
# Current Debian stable uses a newer libraries, so that should be OK.
@pytest.fixture
def tmp_path(tmpdir):
    # Avoid trouble since tmpdir can be a py._path.local.LocalPath
    return Path(str(tmpdir))


@pytest.fixture(autouse=True)
def use_tmp_workdir(tmp_path):
    # Pytest can on some systems provide a Path from pathlib2.  Normalize:
    path = Path(str(tmp_path))
    with workdir(path, mkdir=True):
        yield tmp_path


@pytest.fixture(scope='session')
def tkinter():
    import tkinter
    try:
        tkinter.Tk()
    except tkinter.TclError as err:
        pytest.skip('no tkinter: {}'.format(err))


@pytest.fixture(scope='session')
def plt(tkinter):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    return plt


@pytest.fixture
def figure(plt):
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture(scope='session')
def psycopg2():
    return pytest.importorskip('psycopg2')


@pytest.fixture(scope='session')
def datafiles():
    try:
        import asetest
    except ImportError:
        return {}
    else:
        return asetest.datafiles.paths


@pytest.fixture(scope='session')
def configured_executables():
    return get_testing_executables()


@pytest.fixture(scope='session')
def factories(configured_executables, datafiles, enabled_calculators):
    return Factories(configured_executables, datafiles)


abinit_factory = make_factory_fixture('abinit')
cp2k_factory = make_factory_fixture('cp2k')
dftb_factory = make_factory_fixture('dftb')
espresso_factory = make_factory_fixture('espresso')
gpaw_factory = make_factory_fixture('gpaw')
octopus_factory = make_factory_fixture('octopus')
siesta_factory = make_factory_fixture('siesta')


@pytest.fixture
def factory(request, factories):
    name, kwargs = request.param
    factory = factories[name]
    return CalculatorInputs(factory, kwargs)


def pytest_generate_tests(metafunc):
    from ase.test.factories import parametrize_calculator_tests
    parametrize_calculator_tests(metafunc)

    if 'seed' in metafunc.fixturenames:
        seeds = metafunc.config.getoption('seed')
        if len(seeds) == 0:
            seeds = [0, 1]
        else:
            seeds = list(map(int, seeds))
        metafunc.parametrize('seed', seeds)


class CLI:
    def __init__(self, calculators):
        self.calculators = calculators

    def ase(self, args):
        if isinstance(args, str):
            import shlex
            args = shlex.split(args)

        proc = Popen(['ase', '-T'] + args,
                     stdout=PIPE, stdin=PIPE)
        stdout, _ = proc.communicate(b'')
        status = proc.wait()
        assert status == 0
        return stdout.decode('utf-8')

    def shell(self, command, calculator_name=None):
        if calculator_name is not None:
            self.calculators.require(calculator_name)

        actual_command = ' '.join(command.split('\n')).strip()
        output = check_output(actual_command, shell=True)
        return output.decode()

@pytest.fixture(scope='session')
def datadir():
    from ase.test.testsuite import datadir
    return datadir


@pytest.fixture(scope='session')
def asap3():
    asap3 = pytest.importorskip('asap3')
    return asap3


@pytest.fixture(scope='session')
def cli(calculators):
    return CLI(calculators)


@pytest.fixture(autouse=True)
def arbitrarily_seed_rng(request):
    # We want tests to not use global stuff such as np.random.seed().
    # But they do.
    #
    # So in lieu of (yet) fixing it, we reseed and unseed the random
    # state for every test.  That makes each test deterministic if it
    # uses random numbers without seeding, but also repairs the damage
    # done to global state if it did seed.
    #
    # In order not to generate all the same random numbers in every test,
    # we seed according to a kind of hash:
    import numpy as np
    import zlib
    module_name = request.module
    function_name = request.function.__name__
    hashable_string = f'{module_name}:{function_name}'
    # We use zlib.adler32() rather than hash() because Python randomizes
    # the string hashing at startup for security reasons.
    seed = zlib.adler32(hashable_string.encode('ascii')) % 12345
    # (We should really use the full qualified name of the test method.)
    state = np.random.get_state()
    np.random.seed(seed)
    yield
    np.random.set_state(state)


def pytest_addoption(parser):
    parser.addoption('--calculators', metavar='NAMES', default='',
                     help='comma-separated list of calculators to test')
    parser.addoption('--seed', action='append', default=[],
                     help='add a seed for tests where random number generators'
                          ' are involved. This option can be applied more'
                          ' than once.')
