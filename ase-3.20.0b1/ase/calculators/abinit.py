"""This module defines an ASE interface to ABINIT.

http://www.abinit.org/
"""

import re

import ase.io.abinit as io
from ase.calculators.calculator import FileIOCalculator
from ase.utils import workdir
from subprocess import check_output


def get_abinit_version(command):
    txt = check_output([command, '--version']).decode('ascii')
    # This allows trailing stuff like betas, rc and so
    m = re.match(r'\s*(\d\.\d\.\d)', txt)
    if m is None:
        raise RuntimeError('Cannot recognize abinit version. '
                           'Start of output: {}'
                           .format(txt[:40]))
    return m.group(1)


class Abinit(FileIOCalculator):
    """Class for doing ABINIT calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = Abinit(label='abinit', xc='LDA', ecut=400, toldfe=1e-5)
    """

    implemented_properties = ['energy', 'forces', 'stress', 'magmom']
    ignored_changes = {'pbc'}  # In abinit, pbc is always effectively True.
    command = 'abinit < PREFIX.files > PREFIX.log'
    discard_results_on_any_change = True

    default_parameters = dict(
        xc='LDA',
        smearing=None,
        kpts=None,
        raw=None,
        pps='fhi')

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='abinit', atoms=None, pp_paths=None, **kwargs):
        """Construct ABINIT-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'abinit'.

        Examples
        ========
        Use default values:

        >>> h = Atoms('H', calculator=Abinit(ecut=200, toldfe=0.001))
        >>> h.center(vacuum=3.0)
        >>> e = h.get_potential_energy()

        """

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        self.pp_paths = pp_paths

    def write_input(self, atoms, properties, system_changes):
        """Write input parameters to files-file."""

        with workdir(self.directory, mkdir=True):
            io.write_all_inputs(
                atoms, properties, parameters=self.parameters,
                pp_paths=self.pp_paths,
                label=self.prefix)

    def read(self, label):
        """Read results from ABINIT's text-output file."""
        # XXX I think we should redo the concept of 'restarting'.
        # It makes sense to load a previous calculation as
        #
        #  * static, calculator-independent results
        #  * an actual calculator capable of calculating
        #
        # Either of which is simpler than our current mechanism which
        # implies both at the same time.  Moreover, we don't need
        # something like calc.read(label).
        #
        # What we need for these two purposes is
        #
        #  * calc = MyCalculator.read(basefile)
        #      (or maybe it should return Atoms with calc attached)
        #  * results = read_results(basefile, format='abinit')
        #
        # where basefile determines the file tree.
        FileIOCalculator.read(self, label)
        with workdir(self.directory):
            self.atoms, self.parameters = io.read_ase_and_abinit_inputs(
                self.prefix)
            self.results = io.read_results(self.prefix)

    def read_results(self):
        with workdir(self.directory):
            self.results = io.read_results(self.prefix)

    def get_number_of_iterations(self):
        return self.results['niter']

    def get_electronic_temperature(self):
        return self.results['width']

    def get_number_of_electrons(self):
        return self.results['nelect']

    def get_number_of_bands(self):
        return self.results['nbands']

    def get_k_point_weights(self):
        return self.results['kpoint_weights']

    def get_bz_k_points(self):
        raise NotImplementedError

    def get_ibz_k_points(self):
        return self.results['ibz_kpoints']

    def get_spin_polarized(self):
        return self.results['eigenvalues'].shape[0] == 2

    def get_number_of_spins(self):
        return len(self.results['eigenvalues'])

    def get_fermi_level(self):
        return self.results['fermilevel']

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.results['eigenvalues'][spin, kpt]

    def get_occupations(self, kpt=0, spin=0):
        raise NotImplementedError
