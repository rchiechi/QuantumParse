import numpy as np

from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import full_3x3_to_voigt_6_stress


class LennardJones(Calculator):
    """Lennard Jones potential calculator

    see https://en.wikipedia.org/wiki/Lennard-Jones_potential

    The fundamental definition of this potential is a pairwise energy:

    ``u_ij = 4 epsilon ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )``

    For convenience, we'll use d_ij to refer to "distance vector" and
    ``r_ij`` to refer to "scalar distance". So, with position vectors `r_i`:

    ``r_ij = | r_j - r_i | = | d_ij |``

    The derivative of u_ij is:

    ::

        d u_ij / d r_ij
        = (-24 epsilon / r_ij) ( sigma^12/r_ij^12 - sigma^6/r_ij^6 )

    We can define a pairwise force

    ``f_ij = d u_ij / d d_ij = d u_ij / d r_ij * d_ij / r_ij``

    The terms in front of d_ij are often combined into a "general derivative"

    ``du_ij = (d u_ij / d d_ij) / r_ij``

    The force on an atom is:

    ``f_i = sum_(j != i) f_ij``

    There is some freedom of choice in assigning atomic energies, i.e.
    choosing a way to partition the total energy into atomic contributions.

    We choose a symmetric approach:

    ``u_i = 1/2 sum_(j != i) u_ij``

    The total energy of a system of atoms is then:

    ``u = sum_i u_i = 1/2 sum_(i, j != i) u_ij``

    The stress can be written as ( `(x)` denoting outer product):

    ``sigma = 1/2 sum_(i, j != i) f_ij (x) d_ij = sum_i sigma_i ,``
    with atomic contributions

    ``sigma_i  = 1/2 sum_(j != i) f_ij (x) d_ij``

    Implementation note:

    For computational efficiency, we minimise the number of
    pairwise evaluations, so we iterate once over all the atoms,
    and use NeighbourList with bothways=False. In terms of the
    equations, we therefore effectively restrict the sum over `i != j`
    to `j > i`, and need to manually re-add the "missing" `j < i` contributions.

    Another consideration is the cutoff. We have to ensure that the potential
    goes to zero smoothly as an atom moves across the cutoff threshold,
    otherwise the potential is not continuous. In cases where the cutoff is
    so large that u_ij is very small at the cutoff this is automatically
    ensured, but in general, `u_ij(rc) != 0`.

    In order to catch this case, this implementation shifts the total energy

    ``u'_ij = u_ij - u_ij(rc)``

    which ensures that it is precisely zero at the cutoff.
    However, this means that the energy effectively depends
    on the cutoff, which might lead to unexpected results!

    """

    implemented_properties = ['energy', 'energies', 'forces', 'free_energy']
    implemented_properties += ['stress', 'stresses']  # bulk properties
    default_parameters = {'epsilon': 1.0, 'sigma': 1.0, 'rc': None}
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        sigma: float
          The potential minimum is at  2**(1/6) * sigma, default 1.0
        epsilon: float
          The potential depth, default 1.0
        rc: float, None
          Cut-off for the NeighborList is set to 3 * sigma if None.
          The energy is upshifted to be continuous at rc.
          Default None
        """

        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 3 * self.parameters.sigma

        self.nl = None

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc

        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        # potential value at rc
        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r2 = (distance_vectors ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            c12 = c6 ** 2

            pairwise_energies = 4 * epsilon * (c12 - c6) - e0 * (c6 != 0.0)
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies

            pairwise_forces = (-24 * epsilon * (2 * c12 - c6) / r2)[
                :, np.newaxis
            ] * distance_vectors

            forces[ii] += pairwise_forces.sum(axis=0)
            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product

            # add j < i contributions
            for jj, atom_j in enumerate(neighbors):
                energies[atom_j] += 0.5 * pairwise_energies[jj]
                forces[atom_j] += -pairwise_forces[jj]  # f_ji = - f_ij
                stresses[atom_j] += 0.5 * np.outer(
                    pairwise_forces[jj], distance_vectors[jj]
                )

        # no lattice, no stress
        if self.atoms.number_of_lattice_vectors == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results['stress'] = (
                stresses.sum(axis=0) / self.atoms.get_volume()
            )
            self.results['stresses'] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results['energy'] = energy
        self.results['energies'] = energies

        self.results['free_energy'] = energy

        self.results['forces'] = forces
