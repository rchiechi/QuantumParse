from ase import Atoms
from ase.optimize import BFGS


def test_dftb_relax_dimer(dftb_factory):
    calc = dftb_factory.calc(
        label='dftb',
        Hamiltonian_SCC='No',
        Hamiltonian_PolynomialRepulsive='SetForAll {Yes}',
    )

    atoms = Atoms('Si2', positions=[[5., 5., 5.], [7., 5., 5.]],
                  cell=[12.]*3, pbc=False)
    atoms.calc = calc

    dyn = BFGS(atoms, logfile='-')
    dyn.run(fmax=0.1)

    e = atoms.get_potential_energy()
    assert abs(e - -64.830901) < 1., e
