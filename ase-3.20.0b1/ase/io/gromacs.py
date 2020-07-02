"""
read and write gromacs geometry files
"""

from ase.atoms import Atoms
from ase.parallel import paropen
import numpy as np

from ase.data import atomic_numbers
from ase import units


def read_gromacs(filename):
    """ From:
    http://manual.gromacs.org/current/online/gro.html
    C format
    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
    python: starting from 0, including first excluding last
    0:4 5:10 10:15 15:20 20:28 28:36 36:44 44:52 52:60 60:68

    Import gromacs geometry type files (.gro).
    Reads atom positions,
    velocities(if present) and
    simulation cell (if present)
    """

    atoms = Atoms()
    filed = open(filename, 'r')
    lines = filed.readlines()
    filed.close()
    positions = []
    gromacs_velocities = []
    symbols = []
    tags = []
    gromacs_residuenumbers = []
    gromacs_residuenames = []
    gromacs_atomtypes = []
    sym2tag = {}
    tag = 0
    for line in (lines[2:-1]):
        # print(line[0:5]+':'+line[5:11]+':'+line[11:15]+':'+line[15:20])
        # it is not a good idea to use the split method with gromacs input
        # since the fields are defined by a fixed column number. Therefore,
        # they may not be space between the fields
        # inp = line.split()

        floatvect = float(line[20:28]) * 10.0, \
            float(line[28:36]) * 10.0, \
            float(line[36:44]) * 10.0
        positions.append(floatvect)

        # read velocities
        velocities = np.array([0.0, 0.0, 0.0])
        vx = line[44:52].strip()
        vy = line[52:60].strip()
        vz = line[60:68].strip()

        for iv, vxyz in enumerate([vx, vy, vz]):
            if len(vxyz) > 0:
                try:
                    velocities[iv] = float(vxyz)
                except ValueError:
                    raise ValueError("can not convert velocity to float")
            else:
                velocities = None

        if velocities is not None:
            # velocities from nm/ps to ase units
            velocities *= units.nm / (1000.0 * units.fs)
            gromacs_velocities.append(velocities)

        gromacs_residuenumbers.append(int(line[0:5]))
        gromacs_residuenames.append(line[5:11].strip())

        symbol_read = line[11:16].strip()[0:2]
        if symbol_read not in sym2tag.keys():
            sym2tag[symbol_read] = tag
            tag += 1

        tags.append(sym2tag[symbol_read])
        if symbol_read in atomic_numbers:
            symbols.append(symbol_read)
        elif symbol_read[0] in atomic_numbers:
            symbols.append(symbol_read[0])
        elif symbol_read[-1] in atomic_numbers:
            symbols.append(symbol_read[-1])
        else:
            # not an atomic symbol
            # if we can not determine the symbol, we use
            # the dummy symbol X
            symbols.append("X")

        gromacs_atomtypes.append(line[11:16].strip())

    line = lines[-1]
    atoms = Atoms(symbols, positions, tags=tags)

    if len(gromacs_velocities) == len(atoms):
        atoms.set_velocities(gromacs_velocities)
    elif len(gromacs_velocities) != 0:
        raise ValueError("Some atoms velocities were not specified!")

    if not atoms.has('residuenumbers'):
        atoms.new_array('residuenumbers', gromacs_residuenumbers, int)
        atoms.set_array('residuenumbers', gromacs_residuenumbers, int)
    if not atoms.has('residuenames'):
        atoms.new_array('residuenames', gromacs_residuenames, str)
        atoms.set_array('residuenames', gromacs_residuenames, str)
    if not atoms.has('atomtypes'):
        atoms.new_array('atomtypes', gromacs_atomtypes, str)
        atoms.set_array('atomtypes', gromacs_atomtypes, str)

    # determine PBC and unit cell
    atoms.pbc = False
    inp = lines[-1].split()
    try:
        grocell = list(map(float, inp))
    except ValueError:
        return atoms

    if len(grocell) < 3:
        return atoms

    cell = np.diag(grocell[:3])

    if len(grocell) >= 9:
        cell.flat[[1, 2, 3, 5, 6, 7]] = grocell[3:9]

    atoms.cell = cell * 10.
    atoms.pbc = True
    return atoms


def write_gromacs(fileobj, images):
    """Write gromacs geometry files (.gro).

    Writes:

    * atom positions,
    * velocities (if present, otherwise 0)
    * simulation cell (if present)
    """

    if isinstance(fileobj, str):
        fileobj = paropen(fileobj, 'w')

    if not isinstance(images, (list, tuple)):
        images = [images]

    natoms = len(images[-1])
    try:
        gromacs_residuenames = images[-1].get_array('residuenames')
    except KeyError:
        gromacs_residuenames = []
        for idum in range(natoms):
            gromacs_residuenames.append('1DUM')
    try:
        gromacs_atomtypes = images[-1].get_array('atomtypes')
    except KeyError:
        gromacs_atomtypes = images[-1].get_chemical_symbols()

    try:
        residuenumbers = images[-1].get_array('residuenumbers')
    except KeyError:
        residuenumbers = np.ones(natoms, int)

    pos = images[-1].get_positions()
    pos = pos / 10.0

    vel = images[-1].get_velocities()
    if vel is None:
        vel = pos * 0.0
    else:
        vel *= 1000.0 * units.fs / units.nm

    # No "#" in the first line to prevent read error in VMD
    fileobj.write('A Gromacs structure file written by ASE \n')
    fileobj.write('%5d\n' % len(images[-1]))
    count = 1

    # gromacs line see
    # manual.gromacs.org/documentation/current/user-guide/file-formats.html#gro
    # (EDH: link seems broken as of 2020-02-21)
    #    1WATER  OW1    1   0.126   1.624   1.679  0.1227 -0.0580  0.0434
    for (resnb, resname, atomtype, xyz,
         vxyz) in zip(residuenumbers, gromacs_residuenames,
                      gromacs_atomtypes, pos, vel):

        # THIS SHOULD BE THE CORRECT, PYTHON FORMATTING, EQUIVALENT TO THE
        # C FORMATTING GIVEN IN THE GROMACS DOCUMENTATION:
        # >>> %5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f <<<
        line = ("{0:>5d}{1:<5s}{2:>5s}{3:>5d}{4:>8.3f}{5:>8.3f}{6:>8.3f}"
                "{7:>8.4f}{8:>8.4f}{9:>8.4f}\n"
                .format(resnb, resname, atomtype, count,
                        xyz[0], xyz[1], xyz[2], vxyz[0], vxyz[1], vxyz[2]))

        fileobj.write(line)
        count += 1

    # write box geometry
    if images[-1].get_pbc().any():
        mycell = images[-1].get_cell()
        # gromacs manual (manual.gromacs.org/online/gro.html) says:
        # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        #
        # cell[i,j] is the jth Cartesian coordinate of the ith unit vector
        # cell[0,0] cell[1,1] cell[2,2] v1(x) v2(y) v3(z) fv0[0 1 2]
        # cell[0,1] cell[0,2] cell[1,0] v1(y) v1(z) v2(x) fv1[0 1 2]
        # cell[1,2] cell[2,0] cell[2,1] v2(z) v3(x) v3(y) fv2[0 1 2]
        grocell = mycell.flat[[0, 4, 8, 1, 2, 3, 5, 6, 7]] * 0.1
        fileobj.write(''.join(['{:10.5f}'.format(x) for x in grocell]))
    else:
        # When we do not have a cell, the cell is specified as an empty line
        fileobj.write("\n")
