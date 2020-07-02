"""WSGI Flask-app for browsing a database.

::

    +---------------------+
    | layout.html         |
    | +-----------------+ |    +--------------+
    | | search.html     | |    | layout.html  |
    | |     +           | |    | +---------+  |
    | | table.html ----------->| |row.html |  |
    | |                 | |    | +---------+  |
    | +-----------------+ |    +--------------+
    +---------------------+

You can launch Flask's local webserver like this::

    $ ase db abc.db -w

or this::

    $ python3 -m ase.db.app abc.db

"""

import io
import sys
from typing import Dict, Any, Set
from pathlib import Path

from flask import Flask, render_template, request

from ase.db import connect
from ase.db.core import Database
from ase.formula import Formula
from ase.db.web import create_key_descriptions, Session
from ase.db.row import row2dct, AtomsRow
from ase.db.table import all_columns


root = Path(__file__).parent.parent.parent
app = Flask(__name__, template_folder=str(root))

projects: Dict[str, Dict[str, Any]] = {}


@app.route('/', defaults={'project_name': 'default'})
@app.route('/<project_name>')
@app.route('/<project_name>/')
def search(project_name: str):
    """Search page.

    Contains input form for database query and a table result rows.
    """
    session = Session(project_name)
    project = projects[project_name]
    return render_template(project['search_template'],
                           q=request.args.get('query', ''),
                           p=project,
                           session_id=session.id)


@app.route('/update/<int:sid>/<what>/<x>/')
def update(sid: int, what: str, x: str):
    """Update table of rows inside search page.

    ``what`` must be one of:

    * query: execute query in request.args (x not used)
    * limit: set number of rows to show to x
    * toggle: toggle column x
    * sort: sort after column x
    * page: show page x
    """
    session = Session.get(sid)
    project = projects[session.project_name]
    session.update(what, x, request.args, project)
    table = session.create_table(project['database'],
                                 project['uid_key'],
                                 keys=list(project['key_descriptions']))
    return render_template('ase/db/templates/table.html',
                           t=table,
                           p=project,
                           s=session)


@app.route('/<project_name>/row/<uid>')
def row(project_name: str, uid: str):
    """Show details for one database row."""
    project = projects[project_name]
    uid_key = project['uid_key']
    row = project['database'].get('{uid_key}={uid}'
                                  .format(uid_key=uid_key, uid=uid))
    dct = project['row_to_dict_function'](row, project)
    return render_template(project['row_template'],
                           d=dct, row=row, p=project, uid=uid)


@app.route('/atoms/<project_name>/<int:id>/<type>')
def atoms(project_name: str, id: int, type: str):
    """Return atomic structure as cif, xyz or json."""
    row = projects[project_name]['database'].get(id=id)
    a = row.toatoms()
    if type == 'cif':
        b = io.BytesIO()
        a.pbc = True
        a.write(b, 'cif', wrap=False)
        return b.getvalue(), 200, []

    fd = io.StringIO()
    if type == 'xyz':
        a.write(fd, 'xyz')
    elif type == 'json':
        con = connect(fd, type='json')
        con.write(row,
                  data=row.get('data', {}),
                  **row.get('key_value_pairs', {}))
    else:
        1 / 0

    headers = [('Content-Disposition',
                'attachment; filename="{project_name}-{id}.{type}"'
                .format(project_name=project_name, id=id, type=type))]
    txt = fd.getvalue()
    return txt, 200, headers


@app.route('/gui/<int:id>')
def gui(id: int):
    """Pop ud ase gui window."""
    from ase.visualize import view
    atoms = projects['default']['database'].get_atoms(id)
    view(atoms)
    return '', 204, []


@app.route('/test')
def test():
    from pyjokes import get_joke as j
    return j()


@app.route('/robots.txt')
def robots():
    return ('User-agent: *\n'
            'Disallow: /\n'
            '\n'
            'User-agent: Baiduspider\n'
            'Disallow: /\n'
            '\n'
            'User-agent: SiteCheck-sitecrawl by Siteimprove.com\n'
            'Disallow: /\n',
            200)


def handle_query(args) -> str:
    """Converts request args to ase.db query string."""
    return args['query']


def row_to_dict(row: AtomsRow, project: Dict[str, Any]) -> Dict[str, Any]:
    """Convert row to dict for use in html template."""
    dct = row2dct(row, project['key_descriptions'])
    dct['formula'] = Formula(Formula(row.formula).format('abc')).format('html')
    return dct


def add_project(db: Database) -> None:
    """Add database to projects with name 'default'."""
    all_keys: Set[str] = set()
    for row in db.select(columns=['key_value_pairs'], include_data=False):
        all_keys.update(row._keys)
    kd = {key: (key, '', '') for key in all_keys}
    projects['default'] = {
        'name': 'default',
        'uid_key': 'id',
        'key_descriptions': create_key_descriptions(kd),
        'database': db,
        'row_to_dict_function': row_to_dict,
        'handle_query_function': handle_query,
        'default_columns': all_columns[:],
        'search_template': 'ase/db/templates/search.html',
        'row_template': 'ase/db/templates/row.html'}


if __name__ == '__main__':
    db = connect(sys.argv[1])
    add_project(db)
    app.run(host='0.0.0.0', debug=True)
