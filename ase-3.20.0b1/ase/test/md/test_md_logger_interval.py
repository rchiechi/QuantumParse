import numpy as np
import pytest

from ase.optimize import FIRE, BFGS
from ase.data import s22
from ase.calculators.tip3p import TIP3P
from ase.constraints import FixBondLengths
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io import Trajectory
import ase.units as units


md_cls_and_kwargs = [
    (VelocityVerlet, {}),
    (Langevin, {"temperature": 300 * units.kB, "friction": 0.02}),
]


def make_dimer(constraint=True):
    """Prepare atoms object for testing"""
    dimer = s22.create_s22_system("Water_dimer")

    calc = TIP3P(rc=9.0)
    dimer.calc = calc
    if constraint:
        dimer.constraints = FixBondLengths(
            [(3 * i + j, 3 * i + (j + 1) % 3) for i in range(2)
             for j in [0, 1, 2]]
            )
    return dimer

def fmax(forces):
    return np.sqrt((forces ** 2).sum(axis=1).max())


@pytest.mark.parametrize('cls', [FIRE, BFGS])
def test_optimization_log_and_trajectory_length(cls):
    logfile = 'opt.log'
    trajectory = 'opt.traj'
    atoms = make_dimer()

    print("Testing", str(cls))
    opt = cls(atoms, logfile=logfile, trajectory=trajectory)

    # Run optimizer two times
    opt.run(0.2)
    opt.run(0.1)

    # Test number of lines in log file matches number of frames in trajectory
    with open(logfile, 'rt') as lf:
        lines = [l for l in lf]
    loglines = len(lines)
    print("Number of lines in log file:", loglines)

    with Trajectory(trajectory) as traj:
        trajframes = len(traj)
    print("Number of frames in trajectory:", trajframes)

    # There is a header line in the logfile
    assert loglines == trajframes + 1


@pytest.mark.parametrize('loginterval', [1, 2])
@pytest.mark.parametrize('cls, kwargs', md_cls_and_kwargs)
def test_md_log_and_trajectory_length(cls, kwargs, loginterval):
    timestep = 1 * units.fs
    trajectory = 'md.traj'
    logfile = 'md.log'

    atoms = make_dimer(constraint=False)
    assert not atoms.constraints

    print("Testing", str(cls))
    md = cls(atoms, logfile=logfile, timestep=timestep,
             trajectory=trajectory, loginterval=loginterval, **kwargs)

    # run md two times
    md.run(steps=5)
    md.run(steps=5)

    # Test number of lines in log file matches number of frames in trajectory
    with open(logfile, 'rt') as lf:
        lines = [l for l in lf]
    loglines = len(lines)
    print("Number of lines in log file:", loglines)

    with Trajectory(trajectory) as traj:
        trajframes = len(traj)
    print("Number of frames in trajectory:", trajframes)

    # There is a header line in the logfile
    assert loglines == trajframes + 1
