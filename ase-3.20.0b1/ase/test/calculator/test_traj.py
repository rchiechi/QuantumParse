import pytest

from ase.io import read, write
from ase.build import molecule
from ase.test.factories import ObsoleteFactoryWrapper


parameters = {
    'aims': dict(sc_accuracy_rho=5.e-3, sc_accuracy_forces=1e-4, xc='LDA'),
    'crystal': dict(basis='sto-3g'),
    'gamess_us': dict(label='test_traj'),
    #'elk': dict(tasks=0, rgkmax=5.0, epsengy=1.0, epspot=1.0, tforce=True,
    #            pbc=True),
    'vasp': dict(xc='LDA'),
    'Psi4': {},
}

calc = pytest.mark.calculator
@calc('gpaw', mode='lcao',  basis='sz(dzp)')
@calc('abinit', 'cp2k', 'emt')
def test_h2_traj(factory):
    run(factory)

@pytest.mark.parametrize('name', sorted(parameters))
def test_h2_traj_old(name):
    factory = ObsoleteFactoryWrapper(name)
    run(factory)


def run(factory):
    name = factory.name
    par = parameters.get(name, {})
    h2 = molecule('H2')
    h2.center(vacuum=2.0)
    h2.calc = factory.calc(**par)
    e = h2.get_potential_energy()
    assert not h2.calc.calculation_required(h2, ['energy'])
    f = h2.get_forces()
    assert not h2.calc.calculation_required(h2, ['energy', 'forces'])
    write('h2.traj', h2)
    h2 = read('h2.traj')
    assert abs(e - h2.get_potential_energy()) < 1e-12
    assert abs(f - h2.get_forces()).max() < 1e-12

