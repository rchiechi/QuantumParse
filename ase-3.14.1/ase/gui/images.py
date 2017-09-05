from __future__ import print_function
from math import sqrt

import numpy as np

from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.gui.defaults import read_defaults
from ase.io import read, write, string2index


class Images:
    def __init__(self, images=None):
        self.covalent_radii = covalent_radii.copy()
        self.config = read_defaults()
        self.atom_scale = self.config['radii_scale']
        if images is not None:
            self.initialize(images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index]

    def __iter__(self):
        return iter(self._images)

    # XXXXXXX hack
    # compatibility hacks while allowing variable number of atoms
    def get_dynamic(self, atoms):
        dynamic = np.ones(len(atoms), bool)
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                dynamic[constraint.index] = False
        return dynamic

    def set_dynamic(self, mask, value):
        # Does not make much sense if different images have different
        # atom counts.  Attempts to apply mask to all images,
        # to the extent possible.
        for atoms in self:
            dynamic = self.get_dynamic(atoms)
            dynamic[mask[:len(atoms)]] = value
            atoms.constraints = [c for c in atoms.constraints
                                 if not isinstance(c, FixAtoms)]
            atoms.constraints.append(FixAtoms(mask=~dynamic))

    def get_energy(self, atoms):
        try:
            e = atoms.get_potential_energy() * self.repeat.prod()
        except RuntimeError:
            e = np.nan
        return e

    def get_forces(self, atoms):
        try:
            F = atoms.get_forces(apply_constraint=False)
        except RuntimeError:
            return None
        else:
            return np.tile(F.T, self.repeat.prod()).T

    def initialize(self, images, filenames=None, init_magmom=False):
        nimages = len(images)
        if filenames is None:
            filenames = [None] * nimages
        self.filenames = filenames

        #  The below seems to be about "quaternions"
        if 0:  # XXXXXXXXXXXXXXXXXXXX hasattr(images[0], 'get_shapes'):
            self.Q = np.empty((nimages, self.natoms, 4))
            self.shapes = images[0].get_shapes()
            import os as os
            if os.path.exists('shapes'):
                shapesfile = open('shapes')
                lines = shapesfile.readlines()
                shapesfile.close()
                if '#{type:(shape_x,shape_y,shape_z), .....,}' in lines[0]:
                    shape = eval(lines[1])
                    shapes = []
                    for an in images[0].get_atomic_numbers():
                        shapes.append(shape[an])
                    self.shapes = np.array(shapes)
                else:
                    print('shape file has wrong format')
            else:
                print('no shapesfile found: default shapes were used!')

        else:
            self.shapes = None

        warning = False

        self._images = []

        # Whether length or chemical composition changes:
        self.have_varying_species = False
        for i, atoms in enumerate(images):
            # copy atoms or not?  Not copying allows back-editing,
            # but copying actually forgets things like the attached
            # calculator (might have forces/energies
            self._images.append(atoms)
            self.have_varying_species |= np.any(self[0].numbers !=
                                                atoms.numbers)
            if hasattr(self, 'Q'):
                assert False  # XXX askhl fix quaternions
                self.Q[i] = atoms.get_quaternions()
            if (atoms.pbc != self[0].pbc).any():
                warning = True

        if warning:
            import warnings
            warnings.warn('Not all images have the same boundary conditions!')

        self.maxnatoms = max(len(atoms) for atoms in self)
        self.selected = np.zeros(self.maxnatoms, bool)
        self.selected_ordered = []
        self.visible = np.ones(self.maxnatoms, bool)
        self.nselected = 0
        self.repeat = np.ones(3, int)

    def get_radii(self, atoms):
        radii = np.array([self.covalent_radii[z] for z in atoms.numbers])
        radii *= self.atom_scale
        return radii

    def prepare_new_atoms(self):
        "Marks that the next call to append_atoms should clear the images."
        self.next_append_clears = True

    def append_atoms(self, atoms, filename=None):
        "Append an atoms object to the images already stored."
        self.images.append(atoms)
        self.filenames.append(filename)
        self.initialize(self.images, filenames=self.filenames)
        return

    def read(self, filenames, index=-1, filetype=None):
        images = []
        names = []
        for filename in filenames:
            i = read(filename, index, filetype)

            if not isinstance(i, list):
                i = [i]
            images.extend(i)
            names.extend([filename] * len(i))

        self.initialize(images, names)

    def repeat_unit_cell(self):
        for atoms in self:
            # Get quantities taking into account current repeat():
            ref_energy = self.get_energy(atoms)
            ref_forces = self.get_forces(atoms)
            atoms.calc = SinglePointCalculator(atoms,
                                               energy=ref_energy,
                                               forces=ref_forces)
            atoms.cell *= self.repeat.reshape((3, 1))
        self.repeat = np.ones(3, int)

    def repeat_images(self, repeat):
        repeat = np.array(repeat)
        oldprod = self.repeat.prod()
        images = []
        for atoms in self:
            refcell = atoms.get_cell()
            del atoms[len(atoms) // oldprod:]
            atoms *= repeat
            atoms.cell = refcell
            images.append(atoms)
        self.initialize(images, filenames=self.filenames)
        self.repeat = repeat

    def center(self):
        """Center each image in the existing unit cell, keeping the
        cell constant."""
        for atoms in self:
            atoms.center()

    def graph(self, expr):
        """Routine to create the data in graphs, defined by the
        string expr."""
        import ase.units as units
        code = compile(expr + ',', '<input>', 'eval')

        n = len(self)

        def d(n1, n2):
            return sqrt(((R[n1] - R[n2])**2).sum())

        def a(n1, n2, n3):
            v1 = R[n1] - R[n2]
            v2 = R[n3] - R[n2]
            arg = np.vdot(v1, v2) / (sqrt((v1**2).sum() * (v2**2).sum()))
            if arg > 1.0:
                arg = 1.0
            if arg < -1.0:
                arg = -1.0
            return 180.0 * np.arccos(arg) / np.pi

        def dih(n1, n2, n3, n4):
            # vector 0->1, 1->2, 2->3 and their normalized cross products:
            a = R[n2] - R[n1]
            b = R[n3] - R[n2]
            c = R[n4] - R[n3]
            bxa = np.cross(b, a)
            bxa /= np.sqrt(np.vdot(bxa, bxa))
            cxb = np.cross(c, b)
            cxb /= np.sqrt(np.vdot(cxb, cxb))
            angle = np.vdot(bxa, cxb)
            # check for numerical trouble due to finite precision:
            if angle < -1:
                angle = -1
            if angle > 1:
                angle = 1
            angle = np.arccos(angle)
            if np.vdot(bxa, c) > 0:
                angle = 2 * np.pi - angle
            return angle * 180.0 / np.pi

        # get number of mobile atoms for temperature calculation
        E = np.array([self.get_energy(atoms) for atoms in self])

        s = 0.0

        # Namespace for eval:
        ns = {'E': E,
              'd': d, 'a': a, 'dih': dih}

        data = []
        for i in range(n):
            ns['i'] = i
            ns['s'] = s
            ns['R'] = R = self[i].get_positions()
            ns['V'] = self[i].get_velocities()
            F = self.get_forces(self[i])
            if F is not None:
                ns['F'] = F
            ns['A'] = self[i].get_cell()
            ns['M'] = self[i].get_masses()
            # XXX askhl verify:
            dynamic = self.get_dynamic(self[i])
            if F is not None:
                ns['f'] = f = ((F * dynamic[:, None])**2).sum(1)**.5
                ns['fmax'] = max(f)
                ns['fave'] = f.mean()
            ns['epot'] = epot = E[i]
            ns['ekin'] = ekin = self[i].get_kinetic_energy()
            ns['e'] = epot + ekin
            ndynamic = dynamic.sum()
            ns['T'] = 2.0 * ekin / (3.0 * ndynamic * units.kB)
            data = eval(code, ns)
            if i == 0:
                m = len(data)
                xy = np.empty((m, n))
            xy[:, i] = data
            if i + 1 < n and not self.have_varying_species:
                s += sqrt(((self[i + 1].positions - R)**2).sum())
        return xy

    def write(self, filename, rotations='', show_unit_cell=False, bbox=None,
              **kwargs):
        indices = range(len(self))
        p = filename.rfind('@')
        if p != -1:
            try:
                slice = string2index(filename[p + 1:])
            except ValueError:
                pass
            else:
                indices = indices[slice]
                filename = filename[:p]
                if isinstance(indices, int):
                    indices = [indices]

        images = [self.get_atoms(i) for i in indices]
        if len(filename) > 4 and filename[-4:] in ['.eps', '.png', '.pov']:
            write(filename, images,
                  rotation=rotations, show_unit_cell=show_unit_cell,
                  bbox=bbox, **kwargs)
        else:
            write(filename, images, **kwargs)

    def get_atoms(self, frame, remove_hidden=False):
        atoms = self[frame]
        try:
            E = atoms.get_potential_energy()
        except RuntimeError:
            E = None
        try:
            F = atoms.get_forces()
        except RuntimeError:
            F = None

        # Remove hidden atoms if applicable
        if remove_hidden:
            atoms = atoms[self.visible]
            if F is not None:
                F = F[self.visible]
        atoms.set_calculator(SinglePointCalculator(atoms,
                                                   energy=E,
                                                   forces=F))
        return atoms

    def delete(self, i):
        self.images.pop(i)
        self.filenames.pop(i)
        self.initialize(self.images, self.filenames)

    def aneb(self):
        raise NotImplementedError('broken at the moment')
        n = self.nimages
        assert n % 5 == 0
        levels = n // 5
        n = self.nimages = 2 * levels + 3
        P = np.empty((self.nimages, self.natoms, 3))
        V = np.empty((self.nimages, self.natoms, 3))
        F = np.empty((self.nimages, self.natoms, 3))
        E = np.empty(self.nimages)
        for L in range(levels):
            P[L] = self.P[L * 5]
            P[n - L - 1] = self.P[L * 5 + 4]
            V[L] = self.V[L * 5]
            V[n - L - 1] = self.V[L * 5 + 4]
            F[L] = self.F[L * 5]
            F[n - L - 1] = self.F[L * 5 + 4]
            E[L] = self.E[L * 5]
            E[n - L - 1] = self.E[L * 5 + 4]
        for i in range(3):
            P[levels + i] = self.P[levels * 5 - 4 + i]
            V[levels + i] = self.V[levels * 5 - 4 + i]
            F[levels + i] = self.F[levels * 5 - 4 + i]
            E[levels + i] = self.E[levels * 5 - 4 + i]
        self.P = P
        self.V = V
        self.F = F
        self.E = E

    def interpolate(self, m):
        raise NotImplementedError('broken at the moment')
        assert self.nimages == 2
        self.nimages = 2 + m
        P = np.empty((self.nimages, self.natoms, 3))
        V = np.empty((self.nimages, self.natoms, 3))
        F = np.empty((self.nimages, self.natoms, 3))
        A = np.empty((self.nimages, 3, 3))
        E = np.empty(self.nimages)
        T = np.empty((self.nimages, self.natoms), int)
        D = np.empty((self.nimages, 3))
        P[0] = self.P[0]
        V[0] = self.V[0]
        F[0] = self.F[0]
        A[0] = self.A[0]
        E[0] = self.E[0]
        T[:] = self.T[0]
        for i in range(1, m + 1):
            x = i / (m + 1.0)
            y = 1 - x
            P[i] = y * self.P[0] + x * self.P[1]
            V[i] = y * self.V[0] + x * self.V[1]
            F[i] = y * self.F[0] + x * self.F[1]
            A[i] = y * self.A[0] + x * self.A[1]
            E[i] = y * self.E[0] + x * self.E[1]
            D[i] = y * self.D[0] + x * self.D[1]
        P[-1] = self.P[1]
        V[-1] = self.V[1]
        F[-1] = self.F[1]
        A[-1] = self.A[1]
        E[-1] = self.E[1]
        D[-1] = self.D[1]
        self.P = P
        self.V = V
        self.F = F
        self.A = A
        self.E = E
        self.T = T
        self.D = D
        self.filenames[1:1] = [None] * m
