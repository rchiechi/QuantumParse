import numpy as np
from numpy.testing import assert_allclose
import pytest
from ase.build import bulk
from ase.test.eam_pot import Pt_u3
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS


@pytest.mark.calculator('lammpsrun')
def test_Pt_stress_cellopt(factory):

    # (For now) reuse eam file stuff from other lammps test:
    pot_fn = 'Pt_u3.eam'
    f = open(pot_fn, 'w')
    f.write(Pt_u3)
    f.close()
    params = {}
    params['pair_style'] = 'eam'
    params['pair_coeff'] = ['1 1 {}'.format(pot_fn)]
    with factory.calc(specorder=['Pt'], files=[pot_fn], **params) as calc:
        rng = np.random.RandomState(17)

        atoms = bulk('Pt') * (2, 2, 2)
        atoms.rattle(stdev=0.1)
        atoms.cell += 2 * rng.rand(3, 3)
        atoms.calc = calc

        assert_allclose(atoms.get_stress(), calc.calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)

        opt = BFGS(ExpCellFilter(atoms), trajectory='opt.traj')
        for i, _ in enumerate(opt.irun(fmax=0.001)):
            pass

        cell1_ref = np.array(
            [[0.16524, 3.8999, 3.92855],
             [4.211015, 0.634928, 5.047811],
             [4.429529, 3.293805, 0.447377]]
        )

        assert_allclose(np.asarray(atoms.cell), cell1_ref, atol=3e-4, rtol=3e-4)
        assert_allclose(atoms.get_stress(), calc.calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)

        assert i < 80, 'Expected 59 iterations, got many more: {}'.format(i)
