import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize.precon import PreconLBFGS

import warnings


@pytest.mark.slow
@pytest.mark.parametrize('N', [1, 3])
def test_precon(N):
    a0 = bulk('Cu', cubic=True)
    a0 *= (N, N, N)

    # perturb the atoms
    s = a0.get_scaled_positions()
    s[:, 0] *= 0.995
    a0.set_scaled_positions(s)

    atoms = a0.copy()
    atoms.calc = EMT()

    # check we get a warning about small system
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        opt = PreconLBFGS(atoms, precon="auto")
        if N == 1:
            assert len(w) == 1
            assert "The system is likely too small" in str(w[-1].message)
        else:
            assert len(w) == 0

    # check we get a warning about bad estimate for mu with big cell
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        opt.run(1e-3)
        if N == 1:
            assert len(w) == 0
        else:
            assert len(w) == 1
            assert "capping at mu=1.0" in str(w[-1].message)
