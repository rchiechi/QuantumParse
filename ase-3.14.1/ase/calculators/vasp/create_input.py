from __future__ import print_function
# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import sys
import warnings
from os.path import join, isfile, islink

import numpy as np

from ase.calculators.calculator import kpts2ndarray
from ase.utils import basestring

# Parameters that can be set in INCAR. The values which are None
# are not written and default parameters of VASP are used for them.

float_keys = [
    'aexx',       # Fraction of exact/DFT exchange
    'aggac',      # Fraction of gradient correction to correlation
    'aggax',      # Fraction of gradient correction to exchange
    'aldac',      # Fraction of LDA correlation energy
    'amin',       #
    'amix',       #
    'amix_mag',   #
    'bmix',       # tags for mixing
    'bmix_mag',   #
    'cshift',     # Complex shift for dielectric tensor calculation (LOPTICS)
    'deper',      # relative stopping criterion for optimization of eigenvalue
    'ebreak',     # absolute stopping criterion for optimization of eigenvalues
                  # (EDIFF/N-BANDS/4)
    'efield',     # applied electrostatic field
    'emax',       # energy-range for DOSCAR file
    'emin',       #
    'enaug',      # Density cutoff
    'encut',      # Planewave cutoff
    'encutgw',    # energy cutoff for response function
    'encutfock',  # FFT grid in the HF related routines
    'hfscreen',   # attribute to change from PBE0 to HSE
    'kspacing',   # determines the number of k-points if the KPOINTS
                  # file is not present. KSPACING is the smallest
                  # allowed spacing between k-points in units of
                  # $\AA$^{-1}$.
    'potim',      # time-step for ion-motion (fs)
    'nelect',     # total number of electrons
    'param1',     # Exchange parameter
    'param2',     # Exchange parameter
    'pomass',     # mass of ions in am
    'pstress',    # add this stress to the stress tensor, and energy E = V *
                  # pstress
    'sigma',      # broadening in eV
    'spring',     # spring constant for NEB
    'time',       # special control tag
    'weimin',     # maximum weight for a band to be considered empty
    'zab_vdw',    # vdW-DF parameter
    'zval',       # ionic valence
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'jacobian',   # Weight of lattice to atomic motion
    'ddr',        # (DdR) dimer separation
    'drotmax',    # (DRotMax) number of rotation steps per translation step
    'dfnmin',     # (DFNMin) rotational force below which dimer is not rotated
    'dfnmax',     # (DFNMax) rotational force below which dimer rotation stops
    'stol',       # convergence ratio for minimum eigenvalue
    'sdr',        # finite difference for setting up Lanczos matrix and step
                  # size when translating
    'maxmove',    # Max step for translation for IOPT > 0
    'invcurve',   # Initial curvature for LBFGS (IOPT = 1)
    'timestep',   # Dynamical timestep for IOPT = 3 and IOPT = 7
    'sdalpha',    # Ratio between force and step size for IOPT = 4
    # The next keywords pertain to IOPT = 7 (i.e. FIRE)
    'ftimemax',   # Max time step
    'ftimedec',   # Factor to dec. dt
    'ftimeinc',   # Factor to inc. dt
    'falpha',     # Parameter for velocity damping
    'falphadec',  # Factor to dec. alpha
    'clz',        # electron count for core level shift
    'vdw_radius',  # Cutoff radius for Grimme's DFT-D2 and DFT-D3 and
                   # Tkatchenko and Scheffler's DFT-TS dispersion corrections
    'vdw_scaling',  # Global scaling parameter for Grimme's DFT-D2 dispersion
                    # correction
    'vdw_d',      # Global damping parameter for Grimme's DFT-D2 and Tkatchenko
                  # and Scheffler's DFT-TS dispersion corrections
    'vdw_cnradius',  # Cutoff radius for calculating coordination number in
                    # Grimme's DFT-D3 dispersion correction
    'vdw_s6',     # Damping parameter for Grimme's DFT-D2 and DFT-D3 and
                  # Tkatchenko and Scheffler's DFT-TS dispersion corrections
    'vdw_s8',     # Damping parameter for Grimme's DFT-D3 dispersion correction
    'vdw_sr',     # Scaling parameter for Grimme's DFT-D2 and DFT-D3 and
                  # Tkatchenko and Scheffler's DFT-TS dispersion correction
    'vdw_a1',     # Damping parameter for Grimme's DFT-D3 dispersion correction
    'vdw_a2',     # Damping parameter for Grimme's DFT-D3 dispersion correction
    'eb_k',       # solvent permitivity in Vaspsol
    'tau',        # surface tension parameter in Vaspsol
]

exp_keys = [
    'ediff',      # stopping-criterion for electronic upd.
    'ediffg',     # stopping-criterion for ionic upd.
    'symprec',    # precession in symmetry routines
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'fdstep',     # Finite diference step for IOPT = 1 or 2
]

string_keys = [
    'algo',       # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
    'gga',        # xc-type: PW PB LM or 91 (LDA if not set)
    'metagga',    #
    'prec',       # Precission of calculation (Low, Normal, Accurate)
    'system',     # name of System
    'tebeg',      #
    'teend',      # temperature during run
    'precfock',    # FFT grid in the HF related routines
]

int_keys = [
    'ialgo',      # algorithm: use only 8 (CG) or 48 (RMM-DIIS)
    'ibrion',     # ionic relaxation: 0-MD 1-quasi-New 2-CG
    'icharg',     # charge: 0-WAVECAR 1-CHGCAR 2-atom 10-const
    'idipol',     # monopol/dipol and quadropole corrections
    'images',     # number of images for NEB calculation
    'iniwav',     # initial electr wf. : 0-lowe 1-rand
    'isif',       # calculate stress and what to relax
    'ismear',     # part. occupancies: -5 Blochl -4-tet -1-fermi 0-gaus >0 MP
    'ispin',      # spin-polarized calculation
    'istart',     # startjob: 0-new 1-cont 2-samecut
    'isym',       # symmetry: 0-nonsym 1-usesym 2-usePAWsym
    'iwavpr',     # prediction of wf.: 0-non 1-charg 2-wave 3-comb
    'kpar',       # k-point parallelization paramater
    'ldauprint',  # 0-silent, 1-occ. matrix written to OUTCAR, 2-1+pot. matrix
                  # written
    'ldautype',   # L(S)DA+U: 1-Liechtenstein 2-Dudarev 4-Liechtenstein(LDAU)
    'lmaxmix',    #
    'lorbit',     # create PROOUT
    'maxmix',     #
    'ngx',        # FFT mesh for wavefunctions, x
    'ngxf',       # FFT mesh for charges x
    'ngy',        # FFT mesh for wavefunctions, y
    'ngyf',       # FFT mesh for charges y
    'ngz',        # FFT mesh for wavefunctions, z
    'ngzf',       # FFT mesh for charges z
    'nbands',     # Number of bands
    'nblk',       # blocking for some BLAS calls (Sec. 6.5)
    'nbmod',      # specifies mode for partial charge calculation
    'nelm',       # nr. of electronic steps (default 60)
    'nelmdl',     # nr. of initial electronic steps
    'nelmin',
    'nfree',      # number of steps per DOF when calculting Hessian using
                  # finite differences
    'nkred',      # define sub grid of q-points for HF with
                  # nkredx=nkredy=nkredz
    'nkredx',      # define sub grid of q-points in x direction for HF
    'nkredy',      # define sub grid of q-points in y direction for HF
    'nkredz',      # define sub grid of q-points in z direction for HF
    'nomega',     # number of frequency points
    'nomegar',    # number of frequency points on real axis
    'npar',       # parallelization over bands
    'nsim',       # evaluate NSIM bands simultaneously if using RMM-DIIS
    'nsw',        # number of steps for ionic upd.
    'nupdown',    # fix spin moment to specified value
    'nwrite',     # verbosity write-flag (how much is written)
    'smass',      # Nose mass-parameter (am)
    'vdwgr',      # extra keyword for Andris program
    'vdwrn',      # extra keyword for Andris program
    'voskown',    # use Vosko, Wilk, Nusair interpolation
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'ichain',     # Flag for controlling which method is being used (0=NEB,
                  # 1=DynMat, 2=Dimer, 3=Lanczos) if ichain > 3, then both
                  # IBRION and POTIM are automatically set in the INCAR file
    'iopt',       # Controls which optimizer to use.  for iopt > 0, ibrion = 3
                  # and potim = 0.0
    'snl',        # Maximum dimentionality of the Lanczos matrix
    'lbfgsmem',   # Steps saved for inverse Hessian for IOPT = 1 (LBFGS)
    'fnmin',      # Max iter. before adjusting dt and alpha for IOPT = 7 (FIRE)
    'icorelevel',  # core level shifts
    'clnt',       # species index
    'cln',        # main quantum number of excited core electron
    'cll',        # l quantum number of excited core electron
    'ivdw',       # Choose which dispersion correction method to use
    'nbandsgw',   # Number of bands for GW
    'nbandso',    # Number of occupied bands for electron-hole treatment
    'nbandsv',    # Number of virtual bands for electron-hole treatment
    'ncore',      # Number of cores per band, equal to number of cores divided
                  # by npar
    'mdalgo',     # Determines which MD method of Tomas Bucko to use
    'nedos',      # Number of grid points in DOS
    'turbo',      # Ewald, 0 = Normal, 1 = PME
]

bool_keys = [
    'addgrid',    # finer grid for augmentation charge density
    'kgamma',     # The generated kpoint grid (from KSPACING) is either
                  # centred at the $\Gamma$
                  # point (e.g. includes the $\Gamma$ point)
                  # (KGAMMA=.TRUE.)
    'laechg',     # write AECCAR0/AECCAR1/AECCAR2
    'lasph',      # non-spherical contributions to XC energy (and pot for
                  # VASP.5.X)
    'lasync',     # overlap communcation with calculations
    'lcharg',     #
    'lcorr',      # Harris-correction to forces
    'ldau',       # L(S)DA+U
    'ldiag',      # algorithm: perform sub space rotation
    'ldipol',     # potential correction mode
    'lelf',       # create ELFCAR
    'lepsilon',   # enables to calculate and to print the BEC tensors
    'lhfcalc',    # switch to turn on Hartree Fock calculations
    'loptics',    # calculate the frequency dependent dielectric matrix
    'lpard',      # evaluate partial (band and/or k-point) decomposed charge
                  # density
    'lplane',     # parallelisation over the FFT grid
    'lscalapack',  # switch off scaLAPACK
    'lscalu',     # switch of LU decomposition
    'lsepb',      # write out partial charge of each band separately?
    'lsepk',      # write out partial charge of each k-point separately?
    'lthomas',    #
    'luse_vdw',   # Invoke vdW-DF implementation by Klimes et. al
    'lvdw',   # Invoke DFT-D2 method of Grimme
    'lvhar',      # write Hartree potential to LOCPOT (vasp 5.x)
    'lvtot',      # create WAVECAR/CHGCAR/LOCPOT
    'lwave',      #
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'lclimb',     # Turn on CI-NEB
    'ltangentold',  # Old central difference tangent
    'ldneb',      # Turn on modified double nudging
    'lnebcell',   # Turn on SS-NEB
    'lglobal',    # Optmize NEB globally for LBFGS (IOPT = 1)
    'llineopt',   # Use force based line minimizer for translation (IOPT = 1)
    'lbeefens',   # Switch on print of BEE energy contributions in OUTCAR
    'lbeefbas',   # Switch off print of all BEEs in OUTCAR
    'lcalcpol',   # macroscopic polarization (vasp5.2). 'lcalceps'
    'lcalceps',   # Macroscopic dielectric properties and Born effective charge
                  # tensors (vasp 5.2)

    'lvdw',       # Turns on dispersion correction
    'lvdw_ewald',  # Turns on Ewald summation for Grimme's DFT-D2 and
                   # Tkatchenko and Scheffler's DFT-TS dispersion correction
    'lspectral',  # Use the spectral method to calculate independent particle
                  # polarizability
    'lrpa',       # Include local field effects on the Hartree level only
    'lwannier90',  # Switches on the interface between VASP and WANNIER90
    'lsorbit',    # Enable spin-orbit coupling
    'lsol',       # turn on solvation for Vaspsol
    'lautoscale',  # automatically calculate inverse curvature for VTST LBFGS
    'interactive', # Enables interactive calculation for VaspInteractive
]

list_keys = [
    'dipol',      # center of cell for dipol
    'eint',       # energy range to calculate partial charge for
    'ferwe',      # Fixed band occupation (spin-paired)
    'ferdo',      # Fixed band occupation (spin-plarized)
    'iband',      # bands to calculate partial charge for
    'magmom',     # initial magnetic moments
    'kpuse',      # k-point to calculate partial charge for
    'ropt',       # number of grid points for non-local proj in real space
    'rwigs',      # Wigner-Seitz radii
    'ldauu',      # ldau parameters, has potential to redundant w.r.t. dict
    'ldaul',      # key 'ldau_luj', but 'ldau_luj' can't be read direct from
    'ldauj',      # the INCAR (since it needs to know information about atomic
                  # species. In case of conflict 'ldau_luj' gets written out
                  # when a calculation is set up

    'random_seed',  # List of ints used to seed RNG for advanced MD routines
                    # (Bucko)
    'vdw_c6',     # List of floats of C6 parameters (J nm^6 mol^-1) for each
                  # species (DFT-D2 and DFT-TS)
    'vdw_c6au',   # List of floats of C6 parameters (a.u.) for each species
                  # (DFT-TS)
    'vdw_r0',     # List of floats of R0 parameters (angstroms) for each
                  # species (DFT-D2 and DFT-TS)
    'vdw_r0au',   # List of floats of R0 parameters (a.u.) for each species
                  # (DFT-TS)
    'vdw_alpha',  # List of floats of free-atomic polarizabilities for each
                  # species (DFT-TS)
]

special_keys = [
    'lreal',      # non-local projectors in real space
]

dict_keys = [
    'ldau_luj',   # dictionary with L(S)DA+U parameters, e.g. {'Fe':{'L':2,
                  # 'U':4.0, 'J':0.9}, ...}
]

keys = [
    # 'NBLOCK' and KBLOCK       inner block; outer block
    # 'NPACO' and APACO         distance and nr. of slots for P.C.
    # 'WEIMIN, EBREAK, DEPER    special control tags
]


class GenerateVaspInput(object):
    # Parameters corresponding to 'xc' settings.  This may be modified
    # by the user in-between loading calculators.vasp submodule and
    # instantiating the calculator object with calculators.vasp.Vasp()
    xc_defaults = {
        'lda': {'pp': 'LDA'},
        # GGAs
        'pw91': {'pp': 'GGA', 'gga': '91'},
        'pbe': {'pp': 'PBE', 'gga': 'PE'},
        'pbesol': {'gga': 'PS'},
        'revpbe': {'gga': 'RE'},
        'rpbe': {'gga': 'RP'},
        'am05': {'gga': 'AM'},
        # Meta-GGAs
        'tpss': {'metagga': 'TPSS'},
        'revtpss': {'metagga': 'RTPSS'},
        'm06l': {'metagga': 'M06L'},
        # vdW-DFs
        'vdw-df': {'gga': 'RE', 'luse_vdw': True, 'aggac': 0.},
        'optpbe-vdw': {'gga': 'OR', 'luse_vdw': True, 'aggac': 0.0},
        'optb88-vdw': {'gga': 'BO', 'luse_vdw': True, 'aggac': 0.0,
                       'param1': 1.1 / 6.0, 'param2': 0.22},
        'optb86b-vdw': {'gga': 'MK', 'luse_vdw': True, 'aggac': 0.0,
                        'param1': 0.1234, 'param2': 1.0},
        'vdw-df2': {'gga': 'ML', 'luse_vdw': True, 'aggac': 0.0,
                    'zab_vdw': -1.8867},
        'beef-vdw': {'gga': 'BF', 'luse_vdw': True,
                     'zab_vdw': -1.8867},
        # Hartree-Fock and hybrids
        'hf': {'lhfcalc': True, 'aexx': 1.0, 'aldac': 0.0,
               'aggac': 0.0},
        'b3lyp': {'gga': 'B3', 'lhfcalc': True, 'aexx': 0.2,
                  'aggax': 0.72, 'aggac': 0.81, 'aldac': 0.19},
        'pbe0': {'gga': 'PE', 'lhfcalc': True},
        'hse03': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.3},
        'hse06': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.2},
        'hsesol': {'gga': 'PS', 'lhfcalc': True, 'hfscreen': 0.2}}
    
    # elements which have no-suffix files only
    setups_defaults = {'K':  '_pv',
       'Ca': '_pv',
       'Rb': '_pv',
       'Sr': '_sv',
       'Y':  '_sv',
       'Zr': '_sv',
       'Nb': '_pv',
       'Cs': '_sv',
       'Ba': '_sv',
       'Fr': '_sv',
       'Ra': '_sv',
       'Sc': '_sv'}


    def __init__(self, restart=None):
        self.float_params = {}
        self.exp_params = {}
        self.string_params = {}
        self.int_params = {}
        self.bool_params = {}
        self.list_params = {}
        self.special_params = {}
        self.dict_params = {}
        for key in float_keys:
            self.float_params[key] = None
        for key in exp_keys:
            self.exp_params[key] = None
        for key in string_keys:
            self.string_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in bool_keys:
            self.bool_params[key] = None
        for key in list_keys:
            self.list_params[key] = None
        for key in special_keys:
            self.special_params[key] = None
        for key in dict_keys:
            self.dict_params[key] = None

        # Initialize internal dictionary of input parameters which are
        # not regular VASP keys
        self.input_params = {
            'xc': None,  # Exchange-correlation recipe (e.g. 'B3LYP')
            'pp': None,  # Pseudopotential file (e.g. 'PW91')
            'setups': None,  # Special setups (e.g pv, sv, ...)
            'txt': '-',  # Where to send information
            'kpts': (1, 1, 1),  # k-points
            # Option to use gamma-sampling instead of Monkhorst-Pack:
            'gamma': False,
            # number of points between points in band structures:
            'kpts_nintersections': None,
            # Option to write explicit k-points in units
            # of reciprocal lattice vectors:
            'reciprocal': False}

    def set_xc_params(self, xc):
        """Set parameters corresponding to XC functional"""
        xc = xc.lower()
        if xc is None:
            pass
        elif xc not in self.xc_defaults:
            xc_allowed = ', '.join(self.xc_defaults.keys())
            raise ValueError(
                '{0} is not supported for xc! Supported xc values'
                'are: {1}'.format(xc, xc_allowed))
        else:
            # XC defaults to PBE pseudopotentials
            if 'pp' not in self.xc_defaults[xc]:
                self.set(pp='PBE')
            self.set(**self.xc_defaults[xc])

    def set(self, **kwargs):

        if ((('ldauu' in kwargs) and
             ('ldaul' in kwargs) and
             ('ldauj' in kwargs) and
             ('ldau_luj' in kwargs))):
            raise NotImplementedError(
                'You can either specify ldaul, ldauu, and ldauj OR '
                'ldau_luj. ldau_luj is not a VASP keyword. It is a '
                'dictionary that specifies L, U and J for each '
                'chemical species in the atoms object. '
                'For example for a water molecule:'
                '''ldau_luj={'H':{'L':2, 'U':4.0, 'J':0.9},
                      'O':{'L':2, 'U':4.0, 'J':0.9}}''')

        if 'xc' in kwargs:
            self.set_xc_params(kwargs['xc'])
        for key in kwargs:
            if key in self.float_params:
                self.float_params[key] = kwargs[key]
            elif key in self.exp_params:
                self.exp_params[key] = kwargs[key]
            elif key in self.string_params:
                self.string_params[key] = kwargs[key]
            elif key in self.int_params:
                self.int_params[key] = kwargs[key]
            elif key in self.bool_params:
                self.bool_params[key] = kwargs[key]
            elif key in self.list_params:
                self.list_params[key] = kwargs[key]
            elif key in self.special_params:
                self.special_params[key] = kwargs[key]
            elif key in self.dict_params:
                self.dict_params[key] = kwargs[key]
            elif key in self.input_params:
                self.input_params[key] = kwargs[key]
            else:
                raise TypeError('Parameter not defined: ' + key)

    def check_xc(self):
        """Make sure the calculator has functional & pseudopotentials set up

        If no XC combination, GGA functional or POTCAR type is specified,
        default to PW91. Otherwise, try to guess the desired pseudopotentials.
        """

        p = self.input_params

        # There is no way to correctly guess the desired
        # set of pseudopotentials without 'pp' being set.
        # Usually, 'pp' will be set by 'xc'.
        if 'pp' not in p or p['pp'] is None:
            if self.string_params['gga'] is None:
                p.update({'pp': 'lda'})
            elif self.string_params['gga'] == '91':
                p.update({'pp': 'pw91'})
            elif self.string_params['gga'] == 'PE':
                p.update({'pp': 'pbe'})
            else:
                raise NotImplementedError(
                    "Unable to guess the desired set of pseudopotential"
                    "(POTCAR) files. Please do one of the following: \n"
                    "1. Use the 'xc' parameter to define your XC functional."
                    "These 'recipes' determine the pseudopotential file as "
                    "well as setting the INCAR parameters.\n"
                    "2. Use the 'gga' settings None (default), 'PE' or '91'; "
                    "these correspond to LDA, PBE and PW91 respectively.\n"
                    "3. Set the POTCAR explicitly with the 'pp' flag. The "
                    "value should be the name of a folder on the VASP_PP_PATH"
                    ", and the aliases 'LDA', 'PBE' and 'PW91' are also"
                    "accepted.\n")

        if (p['xc'] is not None and
                p['xc'].lower() == 'lda' and
                p['pp'].lower() != 'lda'):
            warnings.warn("XC is set to LDA, but PP is set to "
                          "{0}. \nThis calculation is using the {0} "
                          "POTCAR set. \n Please check that this is "
                          "really what you intended!"
                          "\n".format(p['pp'].upper()))

    def initialize(self, atoms):
        """Initialize a VASP calculation

        Constructs the POTCAR file (does not actually write it).
        User should specify the PATH
        to the pseudopotentials in VASP_PP_PATH environment variable

        The pseudopotentials are expected to be in:
        LDA:  $VASP_PP_PATH/potpaw/
        PBE:  $VASP_PP_PATH/potpaw_PBE/
        PW91: $VASP_PP_PATH/potpaw_GGA/

        if your pseudopotentials are somewhere else, or named
        differently you may make symlinks at the paths above that
        point to the right place. Alternatively, you may pass the full
        name of a folder on the VASP_PP_PATH to the 'pp' parameter.
        """

        p = self.input_params

        self.check_xc()
        self.all_symbols = atoms.get_chemical_symbols()
        self.natoms = len(atoms)
        self.spinpol = atoms.get_initial_magnetic_moments().any()
        atomtypes = atoms.get_chemical_symbols()

        # Determine the number of atoms of each atomic species
        # sorted after atomic species
        special_setups = []
        symbols = []
        symbolcount = {}
        
        # make sure we find POTCARs for elements which have no-suffix files only
        setups = self.setups_defaults.copy()
        # override with user defined setups 
        if p['setups'] is not None:
            setups.update(p['setups'])

        for m in setups:
            try:
                special_setups.append(int(m))
            except ValueError:
                continue

        for m, atom in enumerate(atoms):
            symbol = atom.symbol
            if m in special_setups:
                pass
            else:
                if symbol not in symbols:
                    symbols.append(symbol)
                    symbolcount[symbol] = 1
                else:
                    symbolcount[symbol] += 1

        # Build the sorting list
        self.sort = []
        self.sort.extend(special_setups)

        for symbol in symbols:
            for m, atom in enumerate(atoms):
                if m in special_setups:
                    pass
                else:
                    if atom.symbol == symbol:
                        self.sort.append(m)
        self.resort = list(range(len(self.sort)))
        for n in range(len(self.resort)):
            self.resort[self.sort[n]] = n
        self.atoms_sorted = atoms[self.sort]

        # Check if the necessary POTCAR files exists and
        # create a list of their paths.
        self.symbol_count = []
        for m in special_setups:
            self.symbol_count.append([atomtypes[m], 1])
        for m in symbols:
            self.symbol_count.append([m, symbolcount[m]])

        sys.stdout.flush()

        # Potpaw folders may be identified by an alias or full name
        for pp_alias, pp_folder in (('lda', 'potpaw'),
                                    ('pw91', 'potpaw_GGA'),
                                    ('pbe', 'potpaw_PBE')):
            if p['pp'].lower() == pp_alias:
                break
        else:
            pp_folder = p['pp']

        if 'VASP_PP_PATH' in os.environ:
            pppaths = os.environ['VASP_PP_PATH'].split(':')
        else:
            pppaths = []
        self.ppp_list = []
        # Setting the pseudopotentials, first special setups and
        # then according to symbols
        for m in special_setups:
            if m in setups:
                special_setup_index = m
            elif str(m) in setups:
                special_setup_index = str(m)
            else:
                raise Exception("Having trouble with special setup index {0}."
                                " Please use an int.".format(m))
            potcar = join(pp_folder,
                          setups[special_setup_index],
                          'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    self.ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    self.ppp_list.append(filename + '.Z')
                    break
            else:
                print('Looking for %s' % potcar)
                raise RuntimeError('No pseudopotential for %s!' % symbol)

        for symbol in symbols:
            try:
                potcar = join(pp_folder, symbol + setups[symbol],
                              'POTCAR')
            except (TypeError, KeyError):
                potcar = join(pp_folder, symbol, 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    self.ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    self.ppp_list.append(filename + '.Z')
                    break
            else:
                print('''Looking for %s
                The pseudopotentials are expected to be in:
                LDA:  $VASP_PP_PATH/potpaw/
                PBE:  $VASP_PP_PATH/potpaw_PBE/
                PW91: $VASP_PP_PATH/potpaw_GGA/''' % potcar)
                raise RuntimeError('No pseudopotential for %s!' % symbol)
        self.converged = None
        self.setups_changed = None

    def write_input(self, atoms, directory='./'):
        from ase.io.vasp import write_vasp
        write_vasp(join(directory, 'POSCAR'),
                   self.atoms_sorted,
                   symbol_count=self.symbol_count)
        self.write_incar(atoms, directory=directory)
        self.write_potcar(directory=directory)
        self.write_kpoints(directory=directory)
        self.write_sort_file(directory=directory)

    def clean(self):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = ['CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR',
                 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR',
                 'OUTCAR', 'PCDAT', 'POTCAR', 'vasprun.xml',
                 'WAVECAR', 'XDATCAR', 'PROCAR', 'ase-sort.dat',
                 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def write_incar(self, atoms, directory='./', **kwargs):
        """Writes the INCAR file."""
        # jrk 1/23/2015 I added this flag because this function has
        # two places where magmoms get written. There is some
        # complication when restarting that often leads to magmom
        # getting written twice. this flag prevents that issue.
        magmom_written = False
        incar = open(join(directory, 'INCAR'), 'w')
        incar.write('INCAR created by Atomic Simulation Environment\n')
        for key, val in self.float_params.items():
            if val is not None:
                incar.write(' %s = %5.6f\n' % (key.upper(), val))
        for key, val in self.exp_params.items():
            if val is not None:
                incar.write(' %s = %5.2e\n' % (key.upper(), val))
        for key, val in self.string_params.items():
            if val is not None:
                incar.write(' %s = %s\n' % (key.upper(), val))
        for key, val in self.int_params.items():
            if val is not None:
                incar.write(' %s = %d\n' % (key.upper(), val))
                if key == 'ichain' and val > 0:
                    incar.write(' IBRION = 3\n POTIM = 0.0\n')
                    for key, val in self.int_params.items():
                        if key == 'iopt' and val is None:
                            print('WARNING: optimization is '
                                  'set to LFBGS (IOPT = 1)')
                            incar.write(' IOPT = 1\n')
                    for key, val in self.exp_params.items():
                        if key == 'ediffg' and val is None:
                            RuntimeError('Please set EDIFFG < 0')
        for key, val in self.list_params.items():
            if val is not None:
                if key in ('dipol', 'eint', 'ropt', 'rwigs'):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.4f ' % x) for x in val]
                # ldau_luj is a dictionary that encodes all the
                # data. It is not a vasp keyword. An alternative to
                # the dictionary is to to use 'ldauu', 'ldauj',
                # 'ldaul', which are vasp keywords.
                elif (key in ('ldauu', 'ldauj') and
                      self.dict_params['ldau_luj'] is None):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.4f ' % x) for x in val]
                elif (key in ('ldaul') and
                      self.dict_params['ldau_luj'] is None):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%d ' % x) for x in val]
                elif key in ('ferwe', 'ferdo'):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.1f ' % x) for x in val]
                elif key in ('iband', 'kpuse'):
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%i ' % x) for x in val]
                elif key == 'magmom':
                    incar.write(' %s = ' % key.upper())
                    magmom_written = True
                    list = [[1, val[0]]]
                    for n in range(1, len(val)):
                        if val[n] == val[n - 1]:
                            list[-1][0] += 1
                        else:
                            list.append([1, val[n]])
                    [incar.write('%i*%.4f ' % (mom[0],
                                               mom[1]))
                     for mom in list]
                incar.write('\n')
        for key, val in self.bool_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if val:
                    incar.write('.TRUE.\n')
                else:
                    incar.write('.FALSE.\n')
        for key, val in self.special_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if key == 'lreal':
                    if isinstance(val, basestring):
                        incar.write(val + '\n')
                    elif isinstance(val, bool):
                        if val:
                            incar.write('.TRUE.\n')
                        else:
                            incar.write('.FALSE.\n')
        for key, val in self.dict_params.items():
            if val is not None:
                if key == 'ldau_luj':
                    llist = ulist = jlist = ''
                    for symbol in self.symbol_count:
                        #  default: No +U
                        luj = val.get(symbol[0], {'L': -1, 'U': 0.0, 'J': 0.0})
                        llist += ' %i' % luj['L']
                        ulist += ' %.3f' % luj['U']
                        jlist += ' %.3f' % luj['J']
                    incar.write(' LDAUL =%s\n' % llist)
                    incar.write(' LDAUU =%s\n' % ulist)
                    incar.write(' LDAUJ =%s\n' % jlist)

        if self.spinpol and not magmom_written:
            if not self.int_params['ispin']:
                incar.write(' ispin = 2\n'.upper())
            # Write out initial magnetic moments
            magmom = atoms.get_initial_magnetic_moments()[self.sort]
            # unpack magmom array if three components specified
            if magmom.ndim > 1:
                magmom = [item for sublist in magmom for item in sublist]
            list = [[1, magmom[0]]]
            for n in range(1, len(magmom)):
                if magmom[n] == magmom[n - 1]:
                    list[-1][0] += 1
                else:
                    list.append([1, magmom[n]])
            incar.write(' magmom = '.upper())
            [incar.write('%i*%.4f ' % (mom[0], mom[1])) for mom in list]
            incar.write('\n')
        incar.close()

    def write_kpoints(self, directory='./', **kwargs):
        """Writes the KPOINTS file."""

        # Don't write anything if KSPACING is being used
        if self.float_params['kspacing'] is not None:
            if self.float_params['kspacing'] > 0:
                return
            else:
                raise ValueError("KSPACING value {0} is not allowable. "
                                 "Please use None or a positive number."
                                 "".format(self.float_params['kspacing']))


        p = self.input_params
        kpoints = open(join(directory, 'KPOINTS'), 'w')
        kpoints.write('KPOINTS created by Atomic Simulation Environment\n')

        if isinstance(p['kpts'], dict):
            p['kpts'] = kpts2ndarray(p['kpts'], atoms=self.atoms)
            p['reciprocal'] = True

        shape = np.array(p['kpts']).shape

        # Wrap scalar in list if necessary
        if shape == ():
            p['kpts'] = [p['kpts']]
            shape = (1, )

        if len(shape) == 1:
            kpoints.write('0\n')
            if shape == (1, ):
                kpoints.write('Auto\n')
            elif p['gamma']:
                kpoints.write('Gamma\n')
            else:
                kpoints.write('Monkhorst-Pack\n')
            [kpoints.write('%i ' % kpt) for kpt in p['kpts']]
            kpoints.write('\n0 0 0\n')
        elif len(shape) == 2:
            kpoints.write('%i \n' % (len(p['kpts'])))
            if p['reciprocal']:
                kpoints.write('Reciprocal\n')
            else:
                kpoints.write('Cartesian\n')
            for n in range(len(p['kpts'])):
                [kpoints.write('%f ' % kpt) for kpt in p['kpts'][n]]
                if shape[1] == 4:
                    kpoints.write('\n')
                elif shape[1] == 3:
                    kpoints.write('1.0 \n')
        kpoints.close()

    def write_potcar(self, suffix="", directory='./'):
        """Writes the POTCAR file."""
        import tempfile
        potfile = open(join(directory, 'POTCAR' + suffix), 'w')
        for filename in self.ppp_list:
            if filename.endswith('R'):
                for line in open(filename, 'r'):
                    potfile.write(line)
            elif filename.endswith('.Z'):
                file_tmp = tempfile.NamedTemporaryFile()
                os.system('gunzip -c %s > %s' % (filename, file_tmp.name))
                for line in file_tmp.readlines():
                    potfile.write(line)
                file_tmp.close()
        potfile.close()

    def write_sort_file(self, directory='./'):
        """Writes a sortings file.

        This file contains information about how the atoms are sorted in
        the first column and how they should be resorted in the second
        column. It is used for restart purposes to get sorting right
        when reading in an old calculation to ASE."""

        file = open(join(directory, 'ase-sort.dat'), 'w')
        for n in range(len(self.sort)):
            file.write('%5i %5i \n' % (self.sort[n], self.resort[n]))

# The below functions are used to restart a calculation and are under early
# constructions

    def read_incar(self, filename='INCAR'):
        """Method that imports settings from INCAR file."""

        self.spinpol = False
        file = open(filename, 'r')
        file.readline()
        lines = file.readlines()
        for line in lines:
            try:
                # Make multiplication, comments, and parameters easier to spot
                line = line.replace("*", " * ")
                line = line.replace("=", " = ")
                line = line.replace("#", "# ")
                data = line.split()
                # Skip empty and commented lines.
                if len(data) == 0:
                    continue
                elif data[0][0] in ['#', '!']:
                    continue
                key = data[0].lower()
                if key in float_keys:
                    self.float_params[key] = float(data[2])
                elif key in exp_keys:
                    self.exp_params[key] = float(data[2])
                elif key in string_keys:
                    self.string_params[key] = str(data[2])
                elif key in int_keys:
                    if key == 'ispin':
                        # JRK added. not sure why we would want to leave ispin
                        # out
                        self.int_params[key] = int(data[2])
                        if int(data[2]) == 2:
                            self.spinpol = True
                    else:
                        self.int_params[key] = int(data[2])
                elif key in bool_keys:
                    if 'true' in data[2].lower():
                        self.bool_params[key] = True
                    elif 'false' in data[2].lower():
                        self.bool_params[key] = False
                elif key in list_keys:
                    list = []
                    if key in ('dipol', 'eint', 'ferwe', 'ferdo',
                               'ropt', 'rwigs',
                               'ldauu', 'ldaul', 'ldauj'):
                        for a in data[2:]:
                            if a in ["!", "#"]:
                                break
                            list.append(float(a))
                    elif key in ('iband', 'kpuse'):
                        for a in data[2:]:
                            if a in ["!", "#"]:
                                break
                            list.append(int(a))
                    self.list_params[key] = list
                    if key == 'magmom':
                        list = []
                        i = 2
                        while i < len(data):
                            if data[i] in ["#", "!"]:
                                break
                            if data[i] == "*":
                                b = list.pop()
                                i += 1
                                for j in range(int(b)):
                                    list.append(float(data[i]))
                            else:
                                list.append(float(data[i]))
                            i += 1
                        self.list_params['magmom'] = list
                        list = np.array(list)
                        if self.atoms is not None:
                            self.atoms.set_initial_magnetic_moments(
                                list[self.resort])
                elif key in special_keys:
                    if key == 'lreal':
                        if 'true' in data[2].lower():
                            self.special_params[key] = True
                        elif 'false' in data[2].lower():
                            self.special_params[key] = False
                        else:
                            self.special_params[key] = data[2]
            except KeyError:
                raise IOError('Keyword "%s" in INCAR is'
                              'not known by calculator.' % key)
            except IndexError:
                raise IOError('Value missing for keyword "%s".' % key)

    def read_kpoints(self, filename='KPOINTS'):
        file = open(filename, 'r')
        lines = file.readlines()
        file.close()
        ktype = lines[2].split()[0].lower()[0]
        if ktype in ['g', 'm', 'a']:
            if ktype == 'g':
                self.set(gamma=True)
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
            elif ktype == 'a':
                kpts = np.array([int(lines[3].split()[i]) for i in range(1)])
            elif ktype == 'm':
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
            self.set(kpts=kpts)
        else:
            if ktype in ['c', 'k']:
                self.set(reciprocal=False)
            else:
                self.set(reciprocal=True)
            kpts = np.array([map(float, line.split()) for line in lines[3:]])
            self.set(kpts=kpts)

    def read_potcar(self):
        """ Read the pseudopotential XC functional from POTCAR file.
        """
        file = open('POTCAR', 'r')
        lines = file.readlines()
        file.close()

        # Search for key 'LEXCH' in POTCAR
        xc_flag = None
        for line in lines:
            key = line.split()[0].upper()
            if key == 'LEXCH':
                xc_flag = line.split()[-1].upper()
                break

        if xc_flag is None:
            raise ValueError('LEXCH flag not found in POTCAR file.')

        # Values of parameter LEXCH and corresponding XC-functional
        xc_dict = {'PE': 'PBE', '91': 'PW91', 'CA': 'LDA'}

        if xc_flag not in xc_dict.keys():
            raise ValueError('Unknown xc-functional flag found in POTCAR,'
                             ' LEXCH=%s' % xc_flag)

        self.input_params['pp'] = xc_dict[xc_flag]

    def todict(self):
        """Returns a dictionary of all parameters 
        that can be used to construct a new calculator object"""
        dict_list = [ 
            'float_params',
            'exp_params',
            'string_params',
            'int_params',
            'bool_params',
            'list_params',
            'special_params',
            'dict_params',
            'input_params'
        ]
        dct = {}
        for item in dict_list:
            dct.update(getattr(self, item))
        for key, val in list(dct.items()):
            if val is None:
                del(dct[key])
        return dct
