""" write gromos96 geometry files
(the exact file format is copied from the freely available
gromacs package, http://www.gromacs.org
its procedure src/gmxlib/confio.c (write_g96_conf)
"""

import numpy as np
from ase.parallel import paropen


def read_gromos(fileobj, index=-1):
    """Read gromos geometry files (.g96).
    Reads:
    atom positions,
    and simulation cell (if present)
    tries to set atom types
    """

    from ase import Atoms
    from ase.data import chemical_symbols

    if isinstance(fileobj, str):
        fileobj = paropen(fileobj, 'r')

    if (index != -1):
        raise ValueError("The Gromos reader only supports index=-1")

    lines = fileobj.readlines()
    read_pos = False
    read_box = False
    tmp_pos = []
    symbols = []
    mycell = None
    for line in lines:
        if (read_pos and ('END' in line)):
            read_pos = False
        if (read_box and ('END' in line)):
            read_box = False
        if read_pos:
            symbol, dummy, x, y, z = line.split()[2:7]
            tmp_pos.append((10 * float(x), 10 * float(y), 10 * float(z)))
            if (len(symbol) != 2):
                symbols.append(symbol[0].lower().capitalize())
            else:
                symbol2 = symbol[0].lower().capitalize() + \
                    symbol[1]
                if symbol2 in chemical_symbols:
                    symbols.append(symbol2)
                else:
                    symbols.append(symbol[0].lower().capitalize())
            if symbols[-1] not in chemical_symbols:
                raise RuntimeError("Symbol '{}' not in chemical symbols"
                                   .format(symbols[-1]))
        if read_box:
            try:
                grocell = list(map(float, line.split()))
            except ValueError:
                pass
            else:
                mycell = np.diag(grocell[:3])
                if len(grocell) >= 9:
                    mycell.flat[[1, 2, 3, 5, 6, 7]] = grocell[3:9]
                mycell *= 10.
        if ('POSITION' in line):
            read_pos = True
        if ('BOX' in line):
            read_box = True

    gmx_system = Atoms(symbols=symbols, positions=tmp_pos, cell=mycell)
    if mycell is not None:
        gmx_system.pbc = True
    return gmx_system


def write_gromos(fileobj, images):
    """Write gromos geometry files (.g96).
    Writes:
    atom positions,
    and simulation cell (if present)
    """

    from ase import units

    if isinstance(fileobj, str):
        fileobj = paropen(fileobj, 'w')

    if not isinstance(images, (list, tuple)):
        images = [images]

    natoms = len(images[-1])
    try:
        gromos_residuenames = images[-1].get_array('residuenames')
    except KeyError:
        gromos_residuenames = []
        for idum in range(natoms):
            gromos_residuenames.append('1DUM')
    try:
        gromos_atomtypes = images[-1].get_array('atomtypes')
    except KeyError:
        gromos_atomtypes = images[-1].get_chemical_symbols()

    pos = images[-1].get_positions()
    pos = pos / 10.0

    vel = images[-1].get_velocities()
    if vel is None:
        vel = pos * 0.0
    else:
        vel *= 1000.0 * units.fs / units.nm

    fileobj.write('TITLE\n')
    fileobj.write('Gromos96 structure file written by ASE \n')
    fileobj.write('END\n')
    fileobj.write('POSITION\n')
    count = 1
    rescount = 0
    oldresname = ''
    for resname, atomtype, xyz in zip(gromos_residuenames,
                                      gromos_atomtypes,
                                      pos):
        if resname != oldresname:
            oldresname = resname
            rescount = rescount + 1
        okresname = resname.lstrip('0123456789 ')
        fileobj.write('%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n' %
                      (rescount, okresname, atomtype, count,
                       xyz[0], xyz[1], xyz[2]))
        count = count + 1
    fileobj.write('END\n')

    if images[-1].get_pbc().any():
        fileobj.write('BOX\n')
        mycell = images[-1].get_cell()
        grocell = mycell.flat[[0, 4, 8, 1, 2, 3, 5, 6, 7]] * 0.1
        fileobj.write(''.join(['{:15.9f}'.format(x) for x in grocell]))
        fileobj.write('\nEND\n')
    return
