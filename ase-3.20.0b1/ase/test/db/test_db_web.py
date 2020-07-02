import pytest

from ase import Atoms
from ase.db import connect


@pytest.fixture(scope='module')
def database(tmp_path_factory):
    with tmp_path_factory.mktemp('dbtest') as dbtest:
        db = connect(dbtest / 'test.db', append=False)
        x = [0, 1, 2]
        t1 = [1, 2, 0]
        t2 = [[2, 3], [1, 1], [1, 0]]

        atoms = Atoms('H2O',
                      [(0, 0, 0),
                       (2, 0, 0),
                       (1, 1, 0)])
        atoms.center(vacuum=5)
        atoms.set_pbc(True)

        db.write(atoms,
                 foo=42.0,
                 bar='abc',
                 data={'x': x,
                       't1': t1,
                       't2': t2})
        db.write(atoms)

        yield db


@pytest.fixture(scope='module')
def client(database):
    pytest.importorskip('flask')
    import ase.db.app as app

    app.add_project(database)
    app.app.testing = True
    return app.app.test_client()


def test_add_columns(database):
    """Test that all keys can be added also for row withous keys."""
    pytest.importorskip('flask')
    from ase.db.web import Session
    from ase.db.app import handle_query
    session = Session('name')
    project = {'default_columns': ['bar'],
               'handle_query_function': handle_query}
    session.update('query', '', {'query': 'id=2'}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert table.columns == []  # selected row doesn't have a foo key
    assert 'foo' in table.addcolumns  # ... but we can add it


def test_db_web(client):
    import io
    from ase.db.web import Session
    from ase.io import read
    c = client
    page = c.get('/').data.decode()
    sid = Session.next_id - 1
    assert 'foo' in page
    for url in [f'/update/{sid}/query/bla/?query=id=1',
                '/default/row/1']:
        resp = c.get(url)
        assert resp.status_code == 200

    for type in ['json', 'xyz', 'cif']:
        url = f'atoms/default/1/{type}'
        resp = c.get(url)
        assert resp.status_code == 200
        atoms = read(io.StringIO(resp.data.decode()), format=type)
        print(atoms.numbers)
        assert (atoms.numbers == [1, 1, 8]).all()
