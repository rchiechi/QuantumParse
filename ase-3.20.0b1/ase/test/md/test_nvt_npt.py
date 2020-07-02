import pytest
from ase import Atoms
from ase.units import fs, kB, GPa
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.utils import seterr
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np


@pytest.fixture(scope='module')
def berendsenparams():
    """Parameters for the two Berendsen algorithms."""
    Bgold = 220.0 * 10000  # Bulk modulus of gold, in bar (1 GPa = 10000 bar)
    nvtparam = dict(temperature=300, taut=1000 * fs)
    nptparam = dict(temperature=300, pressure=5000, taut=1000 * fs, taup=1000 * fs,
                    compressibility=1 / Bgold)
    return dict(nvt=nvtparam, npt=nptparam)


@pytest.fixture(scope='module')
def equilibrated(asap3, berendsenparams):
    """Make an atomic system with equilibrated temperature and pressure."""
    rng = np.random.RandomState(42)
    with seterr(all='raise'):
        print()
        # Must be big enough to avoid ridiculous fluctuations
        atoms = bulk('Au', cubic=True).repeat((3, 3, 3))
        #a[5].symbol = 'Ag'
        print(atoms)
        atoms.calc = asap3.EMT()
        MaxwellBoltzmannDistribution(atoms, 100 * kB, force_temp=True,
                                     rng=rng)
        Stationary(atoms)
        assert abs(atoms.get_temperature() - 100) < 0.0001
        md = NPTBerendsen(atoms, timestep=20 * fs, logfile='-',
                          loginterval=200,
                          **berendsenparams['npt'])
        # Equilibrate for 20 ps
        md.run(steps=1000)
        T = atoms.get_temperature()
        pres = -atoms.get_stress(
            include_ideal_gas=True)[:3].sum() / 3 / GPa * 10000
        print("Temperature: {:.2f} K    Pressure: {:.2f} bar".format(T, pres))
        return atoms


def propagate(atoms, asap3, algorithm, algoargs):
    with seterr(all='raise'):
        print()
        md = algorithm(
            atoms,
            timestep=20 * fs,
            logfile='-',
            loginterval=1000,
            **algoargs)
        # Gather data for 50 ps
        T = []
        p = []
        for i in range(500):
            md.run(5)
            T.append(atoms.get_temperature())
            pres = - atoms.get_stress(include_ideal_gas=True)[:3].sum() / 3
            p.append(pres)
        Tmean = np.mean(T)
        p = np.array(p) / GPa * 10000
        pmean = np.mean(p)
        print('Temperature: {:.2f} K +/- {:.2f} K  (N={})'.format(
            Tmean, np.std(T), len(T)))
        print('Center-of-mass corrected temperature: {:.2f} K'.format(
            Tmean * len(atoms) / (len(atoms) - 1)))
        print('Pressure: {:.2f} bar +/- {:.2f} bar  (N={})'.format(
            pmean, np.std(p), len(p)))
        return Tmean, pmean


def test_nvtberendsen(asap3, equilibrated, berendsenparams):
    t, _ = propagate(Atoms(equilibrated), asap3,
                     NVTBerendsen, berendsenparams['nvt'])
    assert abs(t - berendsenparams['nvt']['temperature']) < 0.5


def test_nptberendsen(asap3, equilibrated, berendsenparams):
    t, p = propagate(Atoms(equilibrated), asap3,
                     NPTBerendsen, berendsenparams['npt'])
    assert abs(t - berendsenparams['npt']['temperature']) < 1.0
    assert abs(p - berendsenparams['npt']['pressure']) < 25.0


def test_npt(asap3, equilibrated, berendsenparams):
    params = berendsenparams['npt']
    # NPT uses different units.  The factor 1.3 is the bulk modulus of gold in
    # ev/Å^3
    t, p = propagate(Atoms(equilibrated), asap3, NPT,
                     dict(temperature=params['temperature'] * kB,
                          externalstress=params['pressure'] / 10000 * GPa,
                          ttime=params['taut'],
                          pfactor=params['taup']**2 * 1.3))
    # Unlike NPTBerendsen, NPT assumes that the center of mass is not
    # thermalized, so the kinetic energy should be 3/2 ' kB * (N-1) * T
    n = len(equilibrated)
    assert abs(t - (n - 1) / n * berendsenparams['npt']['temperature']) < 1.0
    assert abs(p - berendsenparams['npt']['pressure']) < 100.0
