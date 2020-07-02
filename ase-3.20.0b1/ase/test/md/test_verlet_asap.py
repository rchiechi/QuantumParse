from ase.units import fs, kB
from ase.build import bulk
from ase.md import VelocityVerlet
from ase.io import Trajectory, read
from ase.utils import seterr
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def test_verlet_asap(asap3):
    with seterr(all='raise'):
        a = bulk('Au').repeat((2, 2, 2))
        a[5].symbol = 'Ag'
        a.pbc = (True, True, False)
        print(a)
        a.calc = asap3.EMT()
        MaxwellBoltzmannDistribution(a, 300 * kB, force_temp=True)
        Stationary(a)
        assert abs(a.get_temperature() - 300) < 0.0001
        print(a.get_forces())
        md = VelocityVerlet(a, timestep=2 * fs, logfile='-', loginterval=500)
        traj = Trajectory('Au7Ag.traj', 'w', a)
        md.attach(traj.write, 100)
        e0 = a.get_total_energy()
        md.run(steps=10000)
        del traj
        assert abs(a.get_total_energy() - e0) < 0.0001
        assert abs(read('Au7Ag.traj').get_total_energy() - e0) < 0.0001
