from ase.build import diamond100
from ase.optimize import BFGS
from ase.constraints import FixAtoms


def test_dftb_relax_surface(dftb_factory):
    calc = dftb_factory.calc(
        label='dftb',
        kpts=(2, 2, 1),
        Hamiltonian_SCC='Yes',
        Hamiltonian_Filling='Fermi {',
        Hamiltonian_Filling_empty='Temperature [Kelvin] = 500.0',
    )

    a = 5.40632280995384
    atoms = diamond100('Si', (1, 1, 6), a=a, vacuum=6., orthogonal=True,
                       periodic=True)
    atoms.positions[-2:,2] -= 0.2
    atoms.set_constraint(FixAtoms(indices=range(4)))
    atoms.calc = calc

    dyn = BFGS(atoms, logfile='-')
    dyn.run(fmax=0.1)

    e = atoms.get_potential_energy()
    assert abs(e - -214.036907) < 1., e
