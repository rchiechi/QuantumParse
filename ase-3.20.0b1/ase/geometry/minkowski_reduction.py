import itertools
import numpy as np
from ase.utils import pbc2pbc


max_it = 100000    # in practice this is not exceeded


class CycleChecker:

    def __init__(self, d):
        assert d in [2, 3]

        # worst case is the hexagonal cell in 2D and the fcc cell in 3D
        n = {2: 6, 3: 12}[d]

        # max cycle length is total number of primtive cell descriptions
        max_cycle_length = np.prod([n - i for i in range(d)]) * np.prod(d)
        self.visited = np.zeros((max_cycle_length, 3 * d), dtype=int)

    def add_site(self, H):
        # flatten array for simplicity
        H = H.ravel()

        # check if site exists
        found = (self.visited == H).all(axis=1).any()

        # shift all visited sites down and place current site at the top
        self.visited = np.roll(self.visited, 1, axis=0)
        self.visited[0] = H
        return found


def reduction_gauss(B, hu, hv):
    """Calculate a Gauss-reduced lattice basis (2D reduction)."""
    cycle_checker = CycleChecker(d=2)
    u = hu @ B
    v = hv @ B

    for it in range(max_it):
        x = int(round(np.dot(u, v) / np.dot(u, u)))
        hu, hv = hv - x * hu, hu
        u = hu @ B
        v = hv @ B
        site = np.array([hu, hv])
        if np.dot(u, u) >= np.dot(v, v) or cycle_checker.add_site(site):
            return hv, hu

    raise RuntimeError(f"Gaussian basis not found after {max_it} iterations")


def relevant_vectors_2D(u, v):
    cs = np.array([e for e in itertools.product([-1, 0, 1], repeat=2)])
    vs = cs @ [u, v]
    indices = np.argsort(np.linalg.norm(vs, axis=1))[:7]
    return vs[indices], cs[indices]


def closest_vector(t0, u, v):
    t = t0
    a = np.zeros(2, dtype=int)
    rs, cs = relevant_vectors_2D(u, v)

    dprev = float("inf")
    for it in range(max_it):
        ds = np.linalg.norm(rs + t, axis=1)
        index = np.argmin(ds)
        if index == 0 or ds[index] >= dprev:
            return a

        dprev = ds[index]
        r = rs[index]
        kopt = int(round(-np.dot(t, r) / np.dot(r, r)))
        a += kopt * cs[index]
        t = t0 + a[0] * u + a[1] * v

    raise RuntimeError(f"Closest vector not found after {max_it} iterations")


def reduction_full(B):
    """Calculate a Minkowski-reduced lattice basis (3D reduction)."""
    cycle_checker = CycleChecker(d=3)
    H = np.eye(3, dtype=int)
    norms = np.linalg.norm(B, axis=1)

    for it in range(max_it):
        # Sort vectors by norm
        H = H[np.argsort(norms, kind='merge')]

        # Gauss-reduce smallest two vectors
        hw = H[2]
        hu, hv = reduction_gauss(B, H[0], H[1])
        H = np.array([hu, hv, hw])
        R = H @ B

        # Orthogonalize vectors using Gram-Schmidt
        u, v, _ = R
        X = u / np.linalg.norm(u)
        Y = v - X * np.dot(v, X)
        Y /= np.linalg.norm(Y)

        # Find closest vector to last element of R
        pu, pv, pw = R @ np.array([X, Y]).T
        nb = closest_vector(pw, pu, pv)

        # Update basis
        H[2] = [nb[0], nb[1], 1] @ H
        R = H @ B

        norms = np.linalg.norm(R, axis=1)
        if norms[2] >= norms[1] or cycle_checker.add_site(H):
            return R, H

    raise RuntimeError(f"Reduced basis not found after {max_it} iterations")


def minkowski_reduce(cell, pbc=True):
    """Calculate a Minkowski-reduced lattice basis.  The reduced basis
    has the shortest possible vector lengths and has
    norm(a) <= norm(b) <= norm(c).

    Implements the method described in:

    Low-dimensional Lattice Basis Reduction Revisited
    Nguyen, Phong Q. and Stehlé, Damien,
    ACM Trans. Algorithms 5(4) 46:1--46:48, 2009
    https://doi.org/10.1145/1597036.1597050

    Parameters:

    cell: array
        The lattice basis to reduce (in row-vector format).
    pbc: array, optional
        The periodic boundary conditions of the cell (Default `True`).
        If `pbc` is provided, only periodic cell vectors are reduced.

    Returns:

    rcell: array
        The reduced lattice basis.
    op: array
        The unimodular matrix transformation (rcell = op @ cell).
    """
    pbc = pbc2pbc(pbc)
    dim = pbc.sum()

    op = np.eye(3, dtype=int)
    if dim == 2:
        perm = np.argsort(pbc, kind='merge')[::-1]    # stable sort
        pcell = cell[perm][:, perm]

        norms = np.linalg.norm(pcell, axis=1)
        norms[2] = float("inf")
        indices = np.argsort(norms)
        op = op[indices]

        hu, hv = reduction_gauss(pcell, op[0], op[1])

        op[0] = hu
        op[1] = hv
        invperm = np.argsort(perm)
        op = op[invperm][:, invperm]

    elif dim == 3:
        _, op = reduction_full(cell)

    # maintain cell handedness
    if dim == 3:
        if np.sign(np.linalg.det(cell)) != np.sign(np.linalg.det(op @ cell)):
            op = -op
    elif dim == 2:
        index = np.argmin(pbc)
        _cell = cell.copy()
        _cell[index] = (1, 1, 1)
        _rcell = op @ cell
        _rcell[index] = (1, 1, 1)

        if np.sign(np.linalg.det(_cell)) != np.sign(np.linalg.det(_rcell)):
            index = np.argmax(pbc)
            op[index] *= -1

    norms1 = np.sort(np.linalg.norm(cell, axis=1))
    norms2 = np.sort(np.linalg.norm(op @ cell, axis=1))
    if not (norms2 <= norms1 + 1E-12).all():
        raise RuntimeError("Minkowski reduction failed")
    return op @ cell, op
