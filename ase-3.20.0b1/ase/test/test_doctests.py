import doctest
import importlib
from distutils.version import LooseVersion

import pytest
import numpy as np


module_names = """\
ase.atoms
ase.build.tools
ase.cell
ase.collections.collection
ase.dft.kpoints
ase.eos
ase.formula
ase.geometry.cell
ase.geometry.geometry
ase.io.ulm
ase.lattice
ase.phasediagram
ase.spacegroup.spacegroup
ase.spacegroup.xtal
ase.symbols
""".split()


# Fixme: The phasediagram module specifies unknown solver options
@pytest.mark.filterwarnings('ignore:Unknown solver options')
@pytest.mark.parametrize('modname', module_names)
def test_doctest(modname):
    # Older numpies format arrays differently.
    # We use the printoptions contextmanager from numpy 1.15:
    # https://docs.scipy.org/doc/numpy/release.html#id45
    if LooseVersion(np.__version__) < '1.15':
        pytest.skip('need numpy >= 1.15')

    mod = importlib.import_module(modname)
    with np.printoptions(legacy='1.13'):
        print(mod, doctest.testmod(mod, raise_on_error=True))
