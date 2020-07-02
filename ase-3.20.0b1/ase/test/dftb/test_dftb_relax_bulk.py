from ase.build import bulk
from ase.optimize import QuasiNewton
from ase.constraints import ExpCellFilter


def test_dftb_relax_bulk(dftb_factory):
    calc = dftb_factory.calc(
        label='dftb',
        kpts=(3,3,3),
        Hamiltonian_SCC='Yes'
    )

    atoms = bulk('Si')
    atoms.calc = calc

    ecf = ExpCellFilter(atoms)
    dyn = QuasiNewton(ecf)
    dyn.run(fmax=0.01)

    e = atoms.get_potential_energy()
    assert abs(e - -73.150819) < 1., e
