import logging

from ase.optimize.optimize import Optimizer
from ase.units import Ha, Bohr


class Berny(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, dihedral=True):
        """Berny optimizer.

        This is a light ASE wrapper around the ``Berny`` optimizer from
        Pyberny_. It is based on a redundant set of internal coordinates, and as
        such is best suited for optimizing covalently bonded molecules. It does
        not support periodic boundary conditions. You can find more information
        on the Pyberny_ website.

        This optimizer is experimental, and while it can be quite efficient when
        it works, it can sometimes fail entirely. These issues are most likely
        related to almost linear bonding angles. For context, see the
        discussions `here <https://github.com/jhrmnn/pyberny/issues/23>`__ and
        `here <https://gitlab.com/ase/ase/-/merge_requests/889>`__.

        .. _Pyberny: https://github.com/jhrmnn/pyberny

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store internal state. If set, file with
            such a name will be searched and internal state stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        dihedral: boolean
            Defaults to True, which means that dihedral angles will be used.
        """
        from berny import Berny as _Berny, Geometry
        from berny.berny import BernyAdapter

        self._restart_data = None  # Optimizer.__init__() may overwrite
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)
        geom = Geometry(atoms.get_chemical_symbols(), atoms.positions)
        self._berny = _Berny(
            geom,
            debug=True,
            restart=self._restart_data,
            maxsteps=10000000000,  # TODO copied from ase.optimize.Optimizer
            gradientmax=0.,
            gradientrms=0.,
            stepmax=0.,
            steprms=0.,
            dihedral=dihedral,
        )
        # override the default logger to alower per-instance logfile
        log = logging.getLogger('{}.{}'.format(__name__, id(self)))
        log.addHandler(logging.StreamHandler(self.logfile))
        self._berny._log = BernyAdapter(log, self._berny._log_extra)
        # Berny yields the initial geometry the first time because it is
        # typically used as a generator, see berny.optimize()
        next(self._berny)

    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()
        energy = self.atoms.get_potential_energy()
        gradients = -self.atoms.get_forces()
        debug = self._berny.send((energy / Ha, gradients / Ha * Bohr))
        self.dump(debug)
        geom = next(self._berny)
        self.atoms.positions[:] = geom.coords

    def read(self):
        self._restart_data = self.load()
