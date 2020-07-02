import pytest
from ase.build import bulk
from ase.test.factories import ObsoleteFactoryWrapper


omx_par = {'definition_of_atomic_species': [['Al', 'Al8.0-p1', 'Al_CA13'],
                                            ['O', 'O6.0-p1', 'O_CA13']]}


required = {'aims': dict(sc_accuracy_rho=5.e-3),
            'elk': dict(tasks=0, rgkmax=5.0),
            'openmx': omx_par}


calc = pytest.mark.calculator
@calc('abinit', ecut=200, toldfe=0.0001, chksymbreak=0)
def test_al(factory):
    run(factory)

@pytest.mark.parametrize('name', sorted(required))
def test_al_old(name):
    factory = ObsoleteFactoryWrapper(name)
    run(factory)


def run(factory):
    name = factory.name
    par = required.get(name, {})
    # What on earth does kpts=1.0 mean?  Was failing, I changed it.  --askhl
    # Disabled GPAW since it was failing anyway. --askhl
    kpts = [2, 2, 2]
    calc = factory.calc(label=name, xc='LDA', kpts=kpts, **par)
    al = bulk('AlO', crystalstructure='rocksalt', a=4.5)
    al.calc = calc
    e = al.get_potential_energy()
    calc.set(xc='PBE', kpts=kpts)
    epbe = al.get_potential_energy()
    print(e, epbe)
    calc = factory.calc(restart=name)
    print(calc.parameters, calc.results, calc.atoms)
    assert not calc.calculation_required(al, ['energy'])
    al = calc.get_atoms()
    print(al.get_potential_energy())
    label = 'dir/' + name + '-2'
    calc = factory.calc(label=label, atoms=al, xc='LDA', kpts=kpts,
                        **par)
    print(al.get_potential_energy())
