import numpy as np

from ase.parallel import world, broadcast
from ase.geometry import get_distances


def random_unit_vector(rng):
    """Random unit vector equally distributed on the sphere

    Parameter
    ---------
    rng: random number generator object
"""
    ct = -1 + 2 * rng.rand()
    phi = 2 * np.pi * rng.rand()
    st = np.sqrt(1 - ct**2)
    return np.array([st * np.cos(phi), st * np.sin(phi), ct])


def nearest(atoms1, atoms2, cell=None, pbc=None):
    """Return indices of nearest atoms"""
    p1 = atoms1.get_positions()
    p2 = atoms2.get_positions()
    vd_aac, d2_aa = get_distances(p1, p2, cell, pbc)
    return np.argwhere(d2_aa == d2_aa.min())[0]


def attach(atoms1, atoms2, distance, direction=(1, 0, 0),
           maxiter=50, accuracy=1e-5):
    """Attach two structures

    Parameters
    ----------
    atoms1: Atoms
      cell and pbc of this object are used
    atoms2: Atoms
    distance: float
      minimal distance (Angstrom)
    direction: unit vector (3 floats)
      relative direction between center of masses
    maxiter: int
      maximal number of iterations to get required distance, default 100
    accuracy: float
      required accuracy for minimal distance (Angstrom), default 1e-5
    """
    atoms = atoms1.copy()
    atoms2 = atoms2.copy()
    
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    assert len(direction) == 3
    dist2 = distance**2
    
    cm1 = atoms.get_center_of_mass()
    d1max = np.dot(atoms.get_positions() - cm1, direction).max()
    cm2 = atoms2.get_center_of_mass()
    d2max = np.dot(cm2 - atoms2.get_positions(), direction).max()

    # first guess
    atoms2.translate(cm1 - cm2 +
                     direction * (distance + d1max + d2max))
    i1, i2 = nearest(atoms, atoms2, atoms.cell, atoms.pbc)

    for i in range(maxiter):
        dv_c = atoms2[i2].position - atoms[i1].position
        dv2 = (dv_c**2).sum()
            
        vcost = np.dot(dv_c, direction)
        a = np.sqrt(max(0, dist2 - dv2 + vcost**2))
        move = a - vcost
        if abs(move) < accuracy:
            atoms += atoms2
            return atoms
        
        # we need to move
        atoms2.translate(direction * move)
        i1, i2 = nearest(atoms, atoms2, atoms.cell, atoms.pbc)

    raise RuntimeError('attach did not converge')


def attach_randomly(atoms1, atoms2, distance,
                    rng=np.random):
    """Randomly attach two structures with a given minimal distance

    Parameters
    ----------
    atoms1: Atoms object
    atoms2: Atoms object
    distance: float
      Required distance
    rng: random number generator object
      defaults to np.random.RandomState()

    Returns
    -------
    Joined structure as an atoms object.
    """
    atoms2 = atoms2.copy()
    atoms2.rotate('x', random_unit_vector(rng),
                  center=atoms2.get_center_of_mass())
    return attach(atoms1, atoms2, distance,
                  direction=random_unit_vector(rng))


def attach_randomly_and_broadcast(atoms1, atoms2, distance,
                                  rng=np.random,
                                  comm=world):
    """Randomly attach two structures with a given minimal distance
      and ensure that these are distributed.

    Parameters
    ----------
    atoms1: Atoms object
    atoms2: Atoms object
    distance: float
      Required distance
    rng: random number generator object
      defaults to np.random.RandomState()
    comm: communicator to distribute
      Communicator to distribute the structure, default: world

    Returns
    -------
    Joined structure as an atoms object.
    """
    if comm.rank == 0:
        joined = attach_randomly(atoms1, atoms2, distance, rng)
        broadcast(joined, 0, comm=comm)
    else:
        joined = broadcast(None, 0, comm)
    return joined
