import re
import warnings
from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.singlepoint import SinglePointCalculator


_link0_keys = [
    'mem',
    'chk',
    'oldchk',
    'schk',
    'rwf',
    'oldmatrix',
    'oldrawmatrix',
    'int',
    'd2e',
    'save',
    'nosave',
    'errorsave',
    'cpu',
    'nprocshared',
    'gpucpu',
    'lindaworkers',
    'usessh',
    'ssh',
    'debuglinda',
]


_link0_special = [
    'kjob',
    'subst',
]


# Certain problematic methods do not provide well-defined potential energy
# surfaces, because these "composite" methods involve geometry optimization
# and/or vibrational frequency analysis. In addition, the "energy" calculated
# by these methods are typically ZPVE corrected and/or temperature dependent
# free energies.
_problem_methods = [
    'cbs-4m', 'cbs-qb3', 'cbs-apno',
    'g1', 'g2', 'g3', 'g4', 'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp4',
    'w1', 'w1u', 'w1bd', 'w1ro',
]


_xc_to_method = dict(
    pbe='pbepbe',
    pbe0='pbe1pbe',
    hse06='hseh1pbe',
    hse03='ohse2pbe',
    lda='svwn',  # gaussian "knows about" LSDA, but maybe not LDA.
    tpss='tpsstpss',
    revtpss='revtpssrevtpss',
)


def write_gaussian_in(fd, atoms, properties=None, **params):
    params = deepcopy(params)

    if properties is None:
        properties = ['energy']

    # pop method and basis
    method = params.pop('method', None)
    basis = params.pop('basis', None)

    # basisfile, only used if basis=gen
    basisfile = params.pop('basisfile', None)

    # basis can be omitted if basisfile is provided
    if basisfile is not None and basis is None:
        basis = 'gen'

    # determine method from xc if it is provided
    if method is None:
        xc = params.pop('xc', None)
        if xc is None:
            # Default to HF
            method = 'hf'
        else:
            method = _xc_to_method.get(xc.lower(), xc)

    # If the user requests a problematic method, rather than raising an error
    # or proceeding blindly, give the user a warning that the results parsed
    # by ASE may not be meaningful.
    if method.lower() in _problem_methods:
        warnings.warn(
            'The requested method, {}, is a composite method. Composite '
            'methods do not have well-defined potential energy surfaces, '
            'so the energies, forces, and other properties returned by '
            'ASE may not be meaningful, or they may correspond to a '
            'different geometry than the one provided. '
            'Please use these methods with caution.'.format(method)
        )

    # determine charge from initial charges if not passed explicitly
    charge = params.pop('charge', None)
    if charge is None:
        charge = atoms.get_initial_charges().sum()

    # determine multiplicity from initial magnetic moments
    # if not passed explicitly
    mult = params.pop('mult', None)
    if mult is None:
        mult = atoms.get_initial_magnetic_moments().sum() + 1

    # pull out raw list of explicit keywords for backwards compatibility
    extra = params.pop('extra', None)

    # pull out any explicit IOPS
    ioplist = params.pop('ioplist', None)

    # also pull out 'addsec', which e.g. contains modredundant info
    addsec = params.pop('addsec', None)

    # set up link0 arguments
    out = []
    for key in _link0_keys:
        if key not in params:
            continue
        val = params.pop(key)
        if not val or (isinstance(val, str) and key.lower() == val.lower()):
            out.append('%{}'.format(key))
        else:
            out.append('%{}={}'.format(key, val))

    # These link0 keywords have a slightly different syntax
    for key in _link0_special:
        if key not in params:
            continue
        val = params.pop(key)
        if not isinstance(val, str) and isinstance(val, Iterable):
            val = ' '.join(val)
        out.append('%{} L{}'.format(key, val))

    # begin route line
    # note: unlike in old calculator, each route keyword is put on its own
    # line.
    if basis is None:
        out.append('#P {}'.format(method))
    else:
        out.append('#P {}/{}'.format(method, basis))

    for key, val in params.items():
        # assume bare keyword if val is falsey, i.e. '', None, False, etc.
        # also, for backwards compatibility: assume bare keyword if key and
        # val are the same
        if not val or (isinstance(val, str) and key.lower() == val.lower()):
            out.append(key)
        elif isinstance(val, str) and ',' in val:
            out.append('{}({})'.format(key, val))
        elif not isinstance(val, str) and isinstance(val, Iterable):
            out.append('{}({})'.format(key, ','.join(val)))
        else:
            out.append('{}={}'.format(key, val))

    if ioplist is not None:
        out.append('IOP(' + ', '.join(ioplist) + ')')

    if extra is not None:
        out.append(extra)

    # Add 'force' iff the user requested forces, since Gaussian crashes when
    # 'force' is combined with certain other keywords such as opt and irc.
    if 'forces' in properties and 'force' not in params:
        out.append('force')

    # header, charge, and mult
    out += ['', 'Gaussian input prepared by ASE', '',
            '{:.0f} {:.0f}'.format(charge, mult)]

    # atomic positions
    for atom in atoms:
        # this formatting was chosen for backwards compatibility reasons, but
        # it would probably be better to
        # 1) Ensure proper spacing between entries with explicit spaces
        # 2) Use fewer columns for the element
        # 3) Use 'e' (scientific notation) instead of 'f' for positions
        out.append('{:<10s}{:20.10f}{:20.10f}{:20.10f}'
                   .format(atom.symbol, *atom.position))

    # unit cell vectors, in case of periodic boundary conditions
    for ipbc, tv in zip(atoms.pbc, atoms.cell):
        if ipbc:
            out.append('TV {:20.10f}{:20.10f}{:20.10f}'.format(*tv))

    out.append('')

    # if basis='gen', set basisfile. Either give a path to a basisfile, or
    # read in the provided file and paste it verbatim
    if basisfile is not None:
        if basisfile[0] == '@':
            out.append(basisfile)
        else:
            with open(basisfile, 'r') as f:
                out.append(f.read())
    else:
        if basis is not None and basis.lower() == 'gen':
            raise InputError('Please set basisfile')

    if addsec is not None:
        out.append('')
        if isinstance(addsec, str):
            out.append(addsec)
        elif isinstance(addsec, Iterable):
            out += list(addsec)

    out += ['', '']
    fd.write('\n'.join(out))


_re_chgmult = re.compile(r'^\s*[+-]?\d+(?:,\s*|\s+)[+-]?\d+\s*$')
# This is a bit more complex of a regex than we typically want, but it
# can be difficult to determine whether a line contains the charge and
# multiplicity, rather than just another route keyword. By making sure
# that the line contains exactly two *integers*, separated by either
# a comma (and possibly whitespace) or some amount of whitespace, we
# can be more confident that we've actually found the charge and multiplicity.


def read_gaussian_in(fd):
    # TODO: figure out proper way to parse all calculator keywords
    symbols = []
    positions = []
    pbc = np.zeros(3, dtype=bool)
    cell = np.zeros((3, 3))
    npbc = 0
    # We're looking for charge and multiplicity
    for line in fd:
        if _re_chgmult.match(line) is not None:
            tokens = fd.readline().split()
            while tokens:
                symbol = tokens[0]
                pos = list(map(float, tokens[1:4]))
                if symbol.upper() == 'TV':
                    pbc[npbc] = True
                    cell[npbc] = pos
                    npbc += 1
                else:
                    symbols.append(symbol)
                    positions.append(pos)
                tokens = fd.readline().split()
            atoms = Atoms(symbols, positions, pbc=pbc, cell=cell)
            return atoms


# In the interest of using the same RE for both atomic positions and forces,
# we make one of the columns optional. That's because atomic positions have
# 6 columns, while forces only has 5 columns. Otherwise they are very similar.
_re_atom = re.compile(
    r'^\s*\S+\s+(\S+)\s+(?:\S+\s+)?(\S+)\s+(\S+)\s+(\S+)\s*$'
)
_re_forceblock = re.compile(r'^\s*Center\s+Atomic\s+Forces\s+\S+\s*$')
_re_l716 = re.compile(r'^\s*\(Enter .+l716.exe\)$')


def _compare_merge_configs(configs, new):
    """Append new to configs if it contains a new geometry or new data.

    Gaussian sometimes repeats a geometry, for example at the end of an
    optimization, or when a user requests vibrational frequency
    analysis in the same calculation as a geometry optimization.

    In those cases, rather than repeating the structure in the list of
    returned structures, try to merge results if doing so doesn't change
    any previously calculated values. If that's not possible, then create
    a new "image" with the new results.
    """
    if not configs:
        configs.append(new)
        return

    old = configs[-1]

    if old != new:
        configs.append(new)
        return

    oldres = old.calc.results
    newres = new.calc.results
    common_keys = set(oldres).intersection(newres)

    for key in common_keys:
        if np.any(oldres[key] != newres[key]):
            configs.append(new)
            return
    else:
        oldres.update(newres)


def read_gaussian_out(fd, index=-1):
    configs = []
    atoms = None
    energy = None
    dipole = None
    forces = None
    for line in fd:
        line = line.strip()
        if line.startswith(r'1\1\GINC'):
            # We've reached the "archive" block at the bottom, stop parsing
            break

        if (line == 'Input orientation:'
                or line == 'Z-Matrix orientation:'):
            if atoms is not None:
                atoms.calc = SinglePointCalculator(
                    atoms, energy=energy, dipole=dipole, forces=forces,
                )
                _compare_merge_configs(configs, atoms)
            atoms = None
            energy = None
            dipole = None
            forces = None

            numbers = []
            positions = []
            pbc = np.zeros(3, dtype=bool)
            cell = np.zeros((3, 3))
            npbc = 0
            # skip 4 irrelevant lines
            for _ in range(4):
                fd.readline()
            while True:
                match = _re_atom.match(fd.readline())
                if match is None:
                    break
                number = int(match.group(1))
                pos = list(map(float, match.group(2, 3, 4)))
                if number == -2:
                    pbc[npbc] = True
                    cell[npbc] = pos
                    npbc += 1
                else:
                    numbers.append(max(number, 0))
                    positions.append(pos)
            atoms = Atoms(numbers, positions, pbc=pbc, cell=cell)
        elif (line.startswith('Energy=')
                or line.startswith('SCF Done:')):
            # Some semi-empirical methods (Huckel, MINDO3, etc.),
            # or SCF methods (HF, DFT, etc.)
            energy = float(line.split('=')[1].split()[0].replace('D', 'e'))
            energy *= Hartree
        elif (line.startswith('E2 =') or line.startswith('E3 =')
                or line.startswith('E4(') or line.startswith('DEMP5 =')
                or line.startswith('E2(')):
            # MP{2,3,4,5} energy
            # also some double hybrid calculations, like B2PLYP
            energy = float(line.split('=')[-1].strip().replace('D', 'e'))
            energy *= Hartree
        elif line.startswith('Wavefunction amplitudes converged. E(Corr)'):
            # "correlated method" energy, e.g. CCSD
            energy = float(line.split('=')[-1].strip().replace('D', 'e'))
            energy *= Hartree
        elif _re_l716.match(line):
            # Sometimes Gaussian will print "Rotating derivatives to
            # standard orientation" after the matched line (which looks like
            # "(Enter /opt/gaussian/g16/l716.exe)", though the exact path
            # depends on where Gaussian is installed). We *skip* the dipole
            # in this case, because it might be rotated relative to the input
            # orientation (and also it is numerically different even if the
            # standard orientation is the same as the input orientation).
            line = fd.readline().strip()
            if not line.startswith('Dipole'):
                continue
            dip = line.split('=')[1].replace('D', 'e')
            tokens = dip.split()
            dipole = []
            # dipole elements can run together, depending on what method was
            # used to calculate them. First see if there is a space between
            # values.
            if len(tokens) == 3:
                dipole = list(map(float, tokens))
            elif len(dip) % 3 == 0:
                # next, check if the number of tokens is divisible by 3
                nchars = len(dip) // 3
                for i in range(3):
                    dipole.append(float(dip[nchars * i:nchars * (i + 1)]))
            else:
                # otherwise, just give up on trying to parse it.
                dipole = None
                continue
            # this dipole moment is printed in atomic units, e-Bohr
            # ASE uses e-Angstrom for dipole moments.
            dipole = np.array(dipole) * Bohr
        elif _re_forceblock.match(line):
            # skip 2 irrelevant lines
            fd.readline()
            fd.readline()
            forces = []
            while True:
                match = _re_atom.match(fd.readline())
                if match is None:
                    break
                forces.append(list(map(float, match.group(2, 3, 4))))
            forces = np.array(forces) * Hartree / Bohr
    if atoms is not None:
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy, dipole=dipole, forces=forces,
        )
        _compare_merge_configs(configs, atoms)
    return configs[index]
