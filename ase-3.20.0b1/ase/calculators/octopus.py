"""ASE-interface to Octopus.

Ask Hjorth Larsen <asklarsen@gmail.com>
Carlos de Armas

http://tddft.org/programs/octopus/
"""

import os
import numpy as np
from ase.io.octopus.input import (
    process_special_kwargs, kwargs2atoms,
    generate_input, parse_input_file,
    normalize_keywords)
from ase.io.octopus.output import read_eigenvalues_file, read_static_info
from ase.calculators.calculator import (
    FileIOCalculator, EigenvalOccupationMixin, PropertyNotImplementedError)


class OctopusIOError(IOError):
    pass


class Octopus(FileIOCalculator, EigenvalOccupationMixin):
    """Octopus calculator.

    The label is always assumed to be a directory."""

    implemented_properties = ['energy', 'forces',
                              'dipole', 'stress',
                              #'magmom', 'magmoms'
    ]

    command = 'octopus'

    def __init__(self,
                 restart=None,
                 label=None,
                 directory=None,
                 atoms=None,
                 command=None,
                 **kwargs):
        """Create Octopus calculator.

        Label is always taken as a subdirectory.
        Restart is taken to be a label."""

        kwargs.pop('check_keywords', None)  # Ignore old keywords
        kwargs.pop('troublesome_keywords', None)

        if label is not None:
            # restart mechanism in Calculator tends to set the label.
            #import warnings
            #warnings.warn('Please use directory=... instead of label')
            directory = label.rstrip('/')

        if directory is None:
            directory = 'ink-pool'

        self.kwargs = {}

        FileIOCalculator.__init__(self, restart=restart,
                                  ignore_bad_restart_file=False,
                                  directory=directory,
                                  atoms=atoms,
                                  command=command, **kwargs)
        # The above call triggers set() so we can update self.kwargs.

    def set(self, **kwargs):
        """Set octopus input file parameters."""
        kwargs = normalize_keywords(kwargs)
        changes = FileIOCalculator.set(self, **kwargs)
        if changes:
            self.results.clear()
        self.kwargs.update(kwargs)
        # XXX should use 'Parameters' but don't know how

    def get_xc_functional(self):
        """Return the XC-functional identifier.
            'LDA', 'PBE', ..."""
        return self.kwargs.get('xcfunctional', 'LDA')

    def get_bz_k_points(self):
        """Return all the k-points in the 1. Brillouin zone.
        The coordinates are relative to reciprocal latice vectors."""
        # Have not found nice way of extracting this information
        # from Octopus.  Thus unimplemented. -askhl
        raise NotImplementedError

    def get_charges(self, atoms=None):
        raise PropertyNotImplementedError

    def get_fermi_level(self):
        return self.results['efermi']

    def get_potential_energies(self):
        raise PropertyNotImplementedError

    def get_dipole_moment(self, atoms=None):
        if 'dipole' not in self.results:
            msg = ('Dipole moment not calculated.\n'
                   'You may wish to use SCFCalculateDipole=True')
            raise OctopusIOError(msg)
        return self.results['dipole']

    def get_stresses(self):
        raise PropertyNotImplementedError

    def get_number_of_spins(self):
        """Return the number of spins in the calculation.
           Spin-paired calculations: 1, spin-polarized calculation: 2."""
        return 2 if self.get_spin_polarized() else 1

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""

        sc = self.kwargs.get('spincomponents')
        if sc is None or sc == 'unpolarized':
            return False
        elif sc == 'spin_polarized' or sc == 'polarized':
            return True
        else:
            raise NotImplementedError('SpinComponents keyword %s' % sc)

    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone.
        The coordinates are relative to reciprocal latice vectors."""
        return self.results['ibz_k_points']

    def get_k_point_weights(self):
        return self.results['k_point_weights']

    def get_number_of_bands(self):
        return self.results['nbands']

    #def get_magnetic_moments(self, atoms=None):
    #    if self.results['nspins'] == 1:
    #        return np.zeros(len(self.atoms))
    #    return self.results['magmoms'].copy()

    #def get_magnetic_moment(self, atoms=None):
    #    if self.results['nspins'] == 1:
    #        return 0.0
    #    return self.results['magmom']

    def get_occupation_numbers(self, kpt=0, spin=0):
        return self.results['occupations'][kpt, spin].copy()

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.results['eigenvalues'][kpt, spin].copy()

    def _getpath(self, path, check=False):
        path = os.path.join(self.directory, path)
        if check:
            if not os.path.exists(path):
                raise OctopusIOError('No such file or directory: %s' % path)
        return path

    def get_atoms(self):
        return FileIOCalculator.get_atoms(self)

    def read_results(self):
        """Read octopus output files and extract data."""
        fd = open(self._getpath('static/info', check=True))
        self.results.update(read_static_info(fd))

        # If the eigenvalues file exists, we get the eigs/occs from that one.
        # This probably means someone ran Octopus in 'unocc' mode to
        # get eigenvalues (e.g. for band structures), and the values in
        # static/info will be the old (selfconsistent) ones.
        try:
            eigpath = self._getpath('static/eigenvalues', check=True)
        except OctopusIOError:
            pass
        else:
            with open(eigpath) as fd:
                kpts, eigs, occs = read_eigenvalues_file(fd)
                kpt_weights = np.ones(len(kpts))  # XXX ?  Or 1 / len(kpts) ?
            self.results.update(eigenvalues=eigs, occupations=occs,
                                ibz_k_points=kpts,
                                k_point_weights=kpt_weights)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties=properties,
                                     system_changes=system_changes)
        txt = generate_input(atoms, process_special_kwargs(atoms, self.kwargs))
        fd = open(self._getpath('inp'), 'w')
        fd.write(txt)
        fd.close()

    def read(self, directory):
        # XXX label of restart file may not be the same as actual label!
        # This makes things rather tricky.  We first set the label to
        # that of the restart file and arbitrarily expect the remaining code
        # to rectify any consequent inconsistencies.
        self.directory = directory

        inp_path = self._getpath('inp')
        fd = open(inp_path)
        kwargs = parse_input_file(fd)

        self.atoms, kwargs = kwargs2atoms(kwargs)
        self.kwargs.update(kwargs)

        fd.close()
        self.read_results()

    @classmethod
    def recipe(cls, **kwargs):
        from ase import Atoms
        system = Atoms()
        calc = Octopus(CalculationMode='recipe', **kwargs)
        system.calc = calc
        try:
            system.get_potential_energy()
        except OctopusIOError:
            pass
        else:
            raise OctopusIOError('Expected recipe, but found '
                                 'useful physical output!')
