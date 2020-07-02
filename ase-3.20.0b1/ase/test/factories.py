import os
from pathlib import Path
import json

import pytest


def get_testing_executables():
    path = os.environ.get('ASE_EXECUTABLE_CONFIGFILE')
    if path is None:
        path = Path.home() / '.ase' / 'executables.json'
    path = Path(path)
    if path.exists():
        dct = json.loads(path.read_text())
    else:
        dct = {}
    return dct


factory_classes = {}


def factory(name):
    def decorator(cls):
        cls.name = name
        factory_classes[name] = cls
        return cls
    return decorator


def make_factory_fixture(name):
    @pytest.fixture(scope='session')
    def _factory(factories):
        return factories[name]
    _factory.__name__ = '{}_factory'.format(name)
    return _factory



@factory('abinit')
class AbinitFactory:
    def __init__(self, executable, pp_paths):
        self.executable = executable
        self.pp_paths = pp_paths

    def _base_kw(self):
        command = '{} < PREFIX.files > PREFIX.log'.format(self.executable)
        return dict(command=command,
                    pp_paths=self.pp_paths,
                    ecut=150,
                    chksymbreak=0,
                    toldfe=1e-3)

    def calc(self, **kwargs):
        from ase.calculators.abinit import Abinit
        kw = self._base_kw()
        kw.update(kwargs)
        return Abinit(**kw)

    @classmethod
    def fromconfig(cls, config):
        return AbinitFactory(config.executables['abinit'],
                             config.datafiles['abinit'])


@factory('cp2k')
class CP2KFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.cp2k import CP2K
        return CP2K(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return CP2KFactory(config.executables['cp2k'])


@factory('dftb')
class DFTBFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.dftb import Dftb
        from ase.test.testsuite import datadir
        # XXX datafiles should be imported from datafiles project
        # We should include more datafiles for DFTB there, and remove them
        # from ASE's own datadir.
        command = f'{self.executable} > PREFIX.out'
        return Dftb(command=command,
                    slako_dir=str(datadir) + '/',  # XXX not obvious
                    **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['dftb'])


@factory('espresso')
class EspressoFactory:
    def __init__(self, executable, pseudo_dir):
        self.executable = executable
        self.pseudo_dir = pseudo_dir

    def _base_kw(self):
        from ase.units import Ry
        return dict(ecutwfc=300 / Ry)

    def calc(self, **kwargs):
        from ase.calculators.espresso import Espresso
        command = '{} -in PREFIX.pwi > PREFIX.pwo'.format(self.executable)
        pseudopotentials = {}
        for path in self.pseudo_dir.glob('*.UPF'):
            fname = path.name
            # Names are e.g. si_lda_v1.uspp.F.UPF
            symbol = fname.split('_', 1)[0].capitalize()
            pseudopotentials[symbol] = fname

        kw = self._base_kw()
        kw.update(kwargs)
        return Espresso(command=command, pseudo_dir=str(self.pseudo_dir),
                        pseudopotentials=pseudopotentials,
                        **kw)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['espresso']
        assert len(paths) == 1
        return cls(config.executables['espresso'], paths[0])


@factory('gpaw')
class GPAWFactory:
    def calc(self, **kwargs):
        from gpaw import GPAW
        return GPAW(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        import importlib
        spec = importlib.util.find_spec('gpaw')
        # XXX should be made non-pytest dependent
        if spec is None:
            pytest.skip('No gpaw module')
        return cls()


class BuiltinCalculatorFactory:
    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls()


@factory('emt')
class EMTFactory(BuiltinCalculatorFactory):
    pass


@factory('lammpsrun')
class LammpsRunFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.lammpsrun import LAMMPS
        return LAMMPS(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['lammps'])


@factory('octopus')
class OctopusFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.octopus import Octopus
        command = f'{self.executable} > stdout.log'
        return Octopus(command=command, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['octopus'])


@factory('siesta')
class SiestaFactory:
    def __init__(self, executable, pseudo_path):
        self.executable = executable
        self.pseudo_path = pseudo_path

    def calc(self, **kwargs):
        from ase.calculators.siesta import Siesta
        command = '{} < PREFIX.fdf > PREFIX.out'.format(self.executable)
        return Siesta(command=command, pseudo_path=str(self.pseudo_path),
                      **kwargs)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['siesta']
        assert len(paths) == 1
        path = paths[0]
        return cls(config.executables['siesta'], str(path))


class Factories:
    def __init__(self, executables, datafiles):
        assert isinstance(executables, dict), executables
        assert isinstance(datafiles, dict)
        self.executables = executables
        self.datafiles = datafiles
        self._factories = {}

    def __getitem__(self, name):
        # Hmm.  We could also just return a new factory instead of
        # caching them.
        if name not in self._factories:
            cls = factory_classes[name]
            try:
                factory = cls.fromconfig(self)
            except KeyError:
                pytest.skip('Missing configuration for {}'.format(name))
            self._factories[name] = factory
        return self._factories[name]


def parametrize_calculator_tests(metafunc):
    """Parametrize tests using our custom markers.

    We want tests marked with @pytest.mark.calculator(names) to be
    parametrized over the named calculator or calculators."""
    calculator_inputs = []

    for marker in metafunc.definition.iter_markers(name='calculator'):
        calculator_names = marker.args
        kwargs = dict(marker.kwargs)
        marks = kwargs.pop('marks', [])
        for name in calculator_names:
            param = pytest.param((name, kwargs), marks=marks)
            calculator_inputs.append(param)

    if calculator_inputs:
        metafunc.parametrize('factory', calculator_inputs, indirect=True,
                             ids=lambda input: input[0])


class CalculatorInputs:
    def __init__(self, factory, parameters=None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self.factory = factory

    @property
    def name(self):
        return self.factory.name

    def __repr__(self):
        cls = type(self)
        return '{}({}, {})'.format(cls.__name__,
                                   self.name, self.parameters)

    def new(self, **kwargs):
        kw = dict(self.parameters)
        kw.update(kwargs)
        return CalculatorInputs(self.factory, kw)

    def calc(self, **kwargs):
        param = dict(self.parameters)
        param.update(kwargs)
        return self.factory.calc(**param)


class ObsoleteFactoryWrapper:
    # We use this for transitioning older tests to the new framework.
    def __init__(self, name):
        self.name = name

    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)
