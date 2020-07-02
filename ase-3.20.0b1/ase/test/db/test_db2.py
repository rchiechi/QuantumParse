import os

import pytest
import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLength
from ase.db import connect
from ase.io import read
from ase.build import molecule

names = ['testase.json', 'testase.db', 'postgresql', 'mysql', 'mariadb']


@pytest.mark.parametrize('name', names)
def test_db2(name):
    if name == 'postgresql':
        pytest.importorskip('psycopg2')
        if os.environ.get('POSTGRES_DB'):  # gitlab-ci
            name = 'postgresql://ase:ase@postgres:5432/testase'
        else:
            name = os.environ.get('ASE_TEST_POSTGRES_URL')
            if name is None:
                return
    elif name == 'mysql':
        pytest.importorskip('pymysql')
        if os.environ.get('CI_PROJECT_DIR'):  # gitlab-ci
            name = 'mysql://root:ase@mysql:3306/testase_mysql'
        else:
            name = os.environ.get('MYSQL_DB_URL')

        if name is None:
            return
    elif name == 'mariadb':
        pytest.importorskip('pymysql')
        if os.environ.get('CI_PROJECT_DIR'):  # gitlab-ci
            name = 'mariadb://root:ase@mariadb:3306/testase_mysql'
        else:
            name = os.environ.get('MYSQL_DB_URL')

        if name is None:
            return

    c = connect(name)
    print(name, c)

    if 'postgres' in name or 'mysql' in name or 'mariadb' in name:
        c.delete([row.id for row in c.select()])

    id = c.reserve(abc=7)
    c.delete([d.id for d in c.select(abc=7)])
    id = c.reserve(abc=7)
    assert c[id].abc == 7

    a = c.get_atoms(id)
    c.write(Atoms())
    ch4 = molecule('CH4', calculator=EMT())
    ch4.constraints = [FixAtoms(indices=[1]),
                       FixBondLength(0, 2)]
    f1 = ch4.get_forces()
    print(f1)

    c.delete([d.id for d in c.select(C=1)])
    chi = np.array([1 + 0.5j, 0.5])
    if 'db' in name:
        kvp = {'external_tables':
               {'blabla': {'a': 1, 'b': 2, 'c': 3},
                'lala': {'a': 0.01, 'b': 0.02, 'c': 0.0}}}

    else:
        kvp = {'a': 1}

    id = c.write(ch4, key_value_pairs=kvp,
                 data={'1-butyne': 'bla-bla', 'chi': chi})

    row = c.get(id)
    print(row.data['1-butyne'], row.data.chi)
    assert (row.data.chi == chi).all(), (row.data.chi, chi)
    print(row)

    assert len(c.get_atoms(C=1).constraints) == 2

    f2 = c.get(C=1).forces
    assert abs(f2.sum(0)).max() < 1e-14
    f3 = c.get_atoms(C=1).get_forces()
    assert abs(f1 - f3).max() < 1e-14

    a = read(name, index='id={}'.format(id))[0]
    f4 = a.get_forces()
    assert abs(f1 - f4).max() < 1e-14

    with pytest.raises(ValueError):
        c.update(id, abc={'a': 42})

    c.update(id, grr='hmm')
    row = c.get(C=1)
    assert row.id == id
    assert (row.data.chi == chi).all()

    for row in c.select(include_data=False):
        assert len(row.data) == 0

    with pytest.raises(ValueError):
        c.write(ch4, foo=['bar', 2])  # not int, bool, float or str

    with pytest.raises(ValueError):
        c.write(Atoms(), pi='3.14')  # number as a string

    with pytest.raises(ValueError):
        c.write(Atoms(), fmax=0.0)  # reserved word

    with pytest.raises(ValueError):
        c.write(Atoms(), S=42)  # chemical symbol as key

    id = c.write(Atoms(),
                 b=np.bool_(True),
                 i=np.int64(42),
                 n=np.nan,
                 x=np.inf,
                 s='NaN2',
                 A=42)
    row = c[id]
    assert isinstance(row.b, bool)
    assert isinstance(row.i, int)
    assert np.isnan(row.n)
    assert np.isinf(row.x)

    # Make sure deleting a single key works:
    id = c.write(Atoms(), key=7)
    c.update(id, delete_keys=['key'])
    assert 'key' not in c[id]

    e = [row.get('energy') for row in c.select(sort='energy')]
    assert len(e) == 5 and abs(e[0] - 1.991) < 0.0005

    # Test the offset keyword
    ids = [row.get('id') for row in c.select()]
    offset = 2
    assert next(c.select(offset=offset)).id == ids[offset]

