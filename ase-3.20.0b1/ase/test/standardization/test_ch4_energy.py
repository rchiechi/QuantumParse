import pytest
from ase.build import molecule
from ase.calculators.calculator import get_calculator_class
from ase.units import Ry
from ase.utils import workdir


# XXX To be replaced by stuff in ase.test.factories
class CalculatorInputs:
    def __init__(self, name, parameters=None):
        self.name = name
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def __repr__(self):
        cls = type(self)
        return '{}({}, {})'.format(cls.__name__,
                                   self.name, self.parameters)

    def calc(self):
        cls = get_calculator_class(self.name)
        return cls(**self.parameters)


def inputs(name, **parameters):
    return CalculatorInputs(name, parameters)


def _calculate(code, name):
    atoms = molecule(name)
    atoms.center(vacuum=3.5)
    with workdir('test-{}'.format(name), mkdir=True):
        atoms.calc = code.calc()
        return atoms.get_potential_energy()


@pytest.mark.parametrize(
    "spec",
    [
        inputs('openmx', energy_cutoff=350),
        inputs('gamess_us', label='ch4'),
        inputs('gaussian', xc='lda', basis='3-21G'),
    ],
    ids=lambda spec: spec.name)
def test_ch4(tmp_path, spec):
    # XXX Convert to string since pytest can sometimes gives us tmp_path
    # as a pathlib2 path.
    with workdir(str(tmp_path), mkdir=True):
        e_ch4 = _calculate(spec, 'CH4')
        e_c2h2 = _calculate(spec, 'C2H2')
        e_h2 = _calculate(spec, 'H2')
        energy = e_ch4 - 0.5 * e_c2h2 - 1.5 * e_h2
        print(energy)
        ref_energy = -2.8
        assert abs(energy - ref_energy) < 0.3


calc = pytest.mark.calculator
@calc('abinit', ecut=300, chksymbreak=0, toldfe=1e-4)
@calc('cp2k')
@calc('espresso', ecutwfc=300 / Ry)
@calc('gpaw', symmetry='off', mode='pw', txt='gpaw.txt', mixer={'beta': 0.6})
@calc('octopus', Spacing='0.4 * angstrom')
@calc('siesta', marks=pytest.mark.xfail)
def test_ch4_reaction(factory):
    e_ch4 = _calculate(factory, 'CH4')
    e_c2h2 = _calculate(factory, 'C2H2')
    e_h2 = _calculate(factory, 'H2')
    energy = e_ch4 - 0.5 * e_c2h2 - 1.5 * e_h2
    print(energy)
    ref_energy = -2.8
    assert abs(energy - ref_energy) < 0.3
