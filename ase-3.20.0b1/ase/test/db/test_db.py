import pytest
import os
from ase.db import connect

cmd = """
ase -T build H | ase -T run emt -o testase.json &&
ase -T build H2O | ase -T run emt -o testase.json &&
ase -T build O2 | ase -T run emt -o testase.json &&
ase -T build H2 | ase -T run emt -f 0.02 -o testase.json &&
ase -T build O2 | ase -T run emt -f 0.02 -o testase.json &&
ase -T build -x fcc Cu | ase -T run emt -E 5,1 -o testase.json &&
ase -T db -v testase.json natoms=1,Cu=1 --delete --yes &&
ase -T db -v testase.json "H>0" -k hydro=1,abc=42,foo=bar &&
ase -T db -v testase.json "H>0" --delete-keys foo"""


dbnames = [
    'json',
    'db',
    'postgresql',
    'mysql',
    'mariadb'
]



@pytest.mark.slow
@pytest.mark.parametrize('dbname', dbnames)
def test_db(dbname, cli):
    def count(n, *args, **kwargs):
        m = len(list(con.select(columns=['id'], *args, **kwargs)))
        assert m == n, (m, n)

    name = None

    if dbname == 'postgresql':
        pytest.importorskip('psycopg2')
        if os.environ.get('POSTGRES_DB'):  # gitlab-ci
            name = 'postgresql://ase:ase@postgres:5432/testase'
        else:
            name = os.environ.get('ASE_TEST_POSTGRES_URL')
    elif dbname == 'mysql':
        pytest.importorskip('pymysql')
        if os.environ.get('CI_PROJECT_DIR'):  # gitlab-ci
            name = 'mysql://root:ase@mysql:3306/testase_mysql'
        else:
            name = os.environ.get('MYSQL_DB_URL')
    elif dbname == 'mariadb':
        pytest.importorskip('pymysql')
        if os.environ.get('CI_PROJECT_DIR'):  # gitlab-ci
            name = 'mariadb://root:ase@mariadb:3306/testase_mysql'
        else:
            name = os.environ.get('MYSQL_DB_URL')
    elif dbname == 'json':
        name = 'testase.json'
    elif dbname == 'db':
        name = 'testase.db'
    else:
        raise ValueError(f'Bad dbname: {dbname}')

    if name is None:
        pytest.skip('Test requires environment variables')

    if 'postgres' in name or 'mysql' in name or 'mariadb' in name:
        con = connect(name)
        con.delete([row.id for row in con.select()])

    cli.shell(cmd.replace('testase.json', name))

    with connect(name) as con:
        assert con.get_atoms(H=1)[0].magmom == 1
        count(5)
        count(3, 'hydro')
        count(0, 'foo')
        count(3, abc=42)
        count(3, 'abc')
        count(0, 'abc,foo')
        count(3, 'abc,hydro')
        count(0, foo='bar')
        count(1, formula='H2')
        count(1, formula='H2O')
        count(3, 'fmax<0.1')
        count(1, '0.5<mass<1.5')
        count(5, 'energy')
        id = con.reserve(abc=7)
        assert con[id].abc == 7

        for key in ['calculator', 'energy', 'abc', 'name', 'fmax']:
            count(6, sort=key)
            count(6, sort='-' + key)

        con.delete([id])
    cli.shell('ase -T gui --terminal -n 3 {}'.format(name))
