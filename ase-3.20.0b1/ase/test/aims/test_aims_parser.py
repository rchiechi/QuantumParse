# flake8: noqa
import numpy as np
from ase.io import read
from numpy.linalg import norm


def test_parse_socketio():
    write_output_socketio()
    traj = read("aims.out", ":", format="aims-output")

    a1, a2 = traj[0], traj[1]
    f1, f2 = a1.get_forces(), a2.get_forces()
    s1, s2 = a1.get_stress(voigt=False), a2.get_stress(voigt=False)

    assert np.allclose(a1.positions[1, 0], 2.11313574)
    assert np.allclose(a2.positions[1, 0], 2.11313574)

    assert np.allclose(f1[0, 0], -0.108555415821635e-07)
    assert np.allclose(f2[1, 1], 0.167235616064691e-03)

    assert np.allclose(s1[0, 0], 0.00006913)
    assert np.allclose(s2[0, 0], -0.00032660)


def test_run():
    write_output()
    atoms = read("aims.out", format="aims-output")

    # find total energy in aims.out
    key = "| Total energy corrected        :"
    with open("aims.out") as f:
        line = next(l for l in f if key in l)
        ref_energy = float(line.split()[5])

    assert norm(atoms.get_total_energy() - ref_energy) < 1e-12

    # find force in aims.out
    key = "Total atomic forces (unitary forces cleaned) [eV/Ang]:"
    with open("aims.out") as f:
        next(l for l in f if key in l)
        line = next(f)
        ref_force = [float(l) for l in line.split()[2:5]]

    assert norm(atoms.get_forces()[0] - ref_force) < 1e-12

    # find stress in aims.out
    key = "Analytical stress tensor - Symmetrized"
    with open("aims.out") as f:
        next(l for l in f if key in l)
        # scroll to significant lines
        for _ in range(4):
            next(f)
        line = next(f)
        ref_stress = [float(l) for l in line.split()[2:5]]

    assert norm(atoms.get_stress(voigt=False)[0] - ref_stress) < 1e-12

    # find atomic stress in aims.out
    key = "Per atom stress (eV) used for heat flux calculation"
    with open("aims.out") as f:
        next(l for l in f if key in l)
        # scroll to boundary
        next(l for l in f if "-------------" in l)

        line = next(f)
        xx, yy, zz, xy, xz, yz = [float(l) for l in line.split()[2:8]]
        ref_stresses = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]

    assert norm(atoms.get_stresses()[0] - ref_stresses) < 1e-12


def write_output():
    output = "  Basic array size parameters:\n  | Number of species                 :        1\n  | Number of atoms                   :        8\n  | Number of lattice vectors         :        3\n  | Max. basis fn. angular momentum   :        2\n  | Max. atomic/ionic basis occupied n:        3\n  | Max. number of basis fn. types    :        3\n  | Max. radial fns per species/type  :        5\n  | Max. logarithmic grid size        :     1346\n  | Max. radial integration grid size :       42\n  | Max. angular integration grid size:      302\n  | Max. angular grid division number :        8\n  | Radial grid for Hartree potential :     1346\n  | Number of spin channels           :        1\n\n\n  Input geometry:\n  | Unit cell:\n  |        5.42606753        0.00000000        0.00000000\n  |        0.00000000        5.42606753        0.00000000\n  |        0.00000000        0.00000000        5.42606753\n  | Atomic structure:\n  |       Atom                x [A]            y [A]            z [A]\n  |    1: Species Si            0.03431851       -0.09796859        0.09930953\n  |    2: Species Si            5.44231311        2.73920529        2.78205416\n  |    3: Species Si            2.75321969        0.10000784        2.72715717\n  |    4: Species Si            2.73199531        2.68826367       -0.08575931\n  |    5: Species Si            1.34757448        1.42946424        1.25761431\n  |    6: Species Si            1.35486030        4.13154987        4.06589071\n  |    7: Species Si            4.04177845        1.27675199        4.00805480\n  |    8: Species Si            3.99821025        4.01092826        1.42388121\n\n  +-------------------------------------------------------------------+\n  |              Analytical stress tensor - Symmetrized               |\n  |                  Cartesian components [eV/A**3]                   |\n  +-------------------------------------------------------------------+\n  |                x                y                z                |\n  |                                                                   |\n  |  x        -0.01478211      -0.01327277      -0.00355870           |\n  |  y        -0.01327277      -0.01512112      -0.01367280           |\n  |  z        -0.00355870      -0.01367280      -0.01534158           |\n  |                                                                   |\n  |  Pressure:       0.01508160   [eV/A**3]                           |\n  |                                                                   |\n  +-------------------------------------------------------------------+\n\n  ESTIMATED overall HOMO-LUMO gap:      0.21466369 eV between HOMO at k-point 1 and LUMO at k-point 1\n\n  Energy and forces in a compact form:\n  | Total energy uncorrected      :         -0.630943948216411E+05 eV\n  | Total energy corrected        :         -0.630943948568205E+05 eV  <-- do not rely on this value for anything but (periodic) metals\n  | Electronic free energy        :         -0.630943948919999E+05 eV\n  Total atomic forces (unitary forces cleaned) [eV/Ang]:\n  |   1         -0.104637839735875E+01          0.500412824184706E+00         -0.439789552504239E+00\n  |   2         -0.155820611394662E+00         -0.476557335046913E+00         -0.655396151432312E+00\n  |   3         -0.193381405004926E+01         -0.122454085397628E+01         -0.169259060410046E+01\n  |   4          0.404969041951871E-01          0.457139849737633E+00         -0.128445757910440E+00\n  |   5          0.109984435024380E-01         -0.165609149153507E+00          0.114351292468512E+01\n  |   6          0.663029766776301E+00         -0.814079627100908E-01          0.384378715376525E-04\n  |   7          0.213211510059627E+01          0.918575437083381E+00          0.189666102862743E+01\n  |   8          0.289372843732474E+00          0.719871898810707E-01         -0.123990325236629E+00\n\n\n    - Per atom stress (eV) used for heat flux calculation:\n        Atom   | Stress components (1,1), (2,2), (3,3), (1,2), (1,3), (2,3)\n      -------------------------------------------------------------------\n             1 |     0.9843662637E-01   -0.1027274769E+00    0.7237959330E-01   -0.3532042840E+00    0.2563317062E+00   -0.3642257991E+00\n             2 |     0.1244911861E+00   -0.4107147872E-01   -0.1084329966E+00    0.1201650287E+00   -0.1716383020E+00   -0.4669712541E-01\n             3 |    -0.1019986539E+01   -0.7054557814E+00   -0.8410240482E+00   -0.3714228752E+00   -0.4921256188E+00   -0.7970402772E+00\n             4 |    -0.5372048581E+00   -0.2498902919E+00   -0.2260340202E+00   -0.4368600591E+00    0.8622059429E-01    0.9182206824E-01\n             5 |    -0.3268304136E-01   -0.1853638313E+00    0.8046857169E-01   -0.3825550863E+00    0.3088175411E+00   -0.2399437437E+00\n             6 |    -0.2682129292E+00   -0.3832959470E+00   -0.5895171406E+00   -0.8151368635E-02    0.5046578049E-01   -0.6756388823E+00\n             7 |    -0.6970248515E+00   -0.6819450154E+00   -0.9123466446E+00   -0.5254451278E+00   -0.5070403877E+00   -0.6281674944E+00\n             8 |    -0.2933806554E-01   -0.6593089867E-01    0.7360641037E-01   -0.1629233327E+00   -0.9955320981E-01    0.4755870988E+00\n      -------------------------------------------------------------------\n\n\n          Have a nice day.\n------------------------------------------------------------\n"

    with open("aims.out", "w") as f:
        f.write(output)


def write_output_socketio():
    output = """
  MPI-parallelism will be employed.
------------------------------------------------------------
          Invoking FHI-aims ...

          When using FHI-aims, please cite the following reference:

          Volker Blum, Ralf Gehrke, Felix Hanke, Paula Havu,
          Ville Havu, Xinguo Ren, Karsten Reuter, and Matthias Scheffler,
          'Ab Initio Molecular Simulations with Numeric Atom-Centered Orbitals',
          Computer Physics Communications 180, 2175-2196 (2009)

          For any questions about FHI-aims, please visit the aimsclub website
          with its forums and wiki. Contributions to both the forums and the
          wiki are warmly encouraged - they are for you, and everyone is welcome there.

------------------------------------------------------------



  Date     :  20200311, Time     :  204314.253
  Time zero on CPU 1             :   0.194170000000000E-01  s.
  Internal wall clock time zero  :           353191394.253  s.

  FHI-aims created a unique identifier for this run for later identification
  aims_uuid : CF4170FC-8FB7-4BC2-A03C-9EC1F9140948

  Build configuration of the current instance of FHI-aims
  -------------------------------------------------------
  FHI-aims version      : 190813
  Commit number         : 6bedc122e
  CMake host system     : Linux-4.15.0-58-generic
  CMake version         : 3.5.1
  Fortran compiler      : /home/knoop/local/openmpi-3.0.0/build/bin/mpifort (Intel) version 17.0.1.20161005
  Fortran compiler flags: -O3 -ip -fp-model precise
  C compiler            : /usr/bin/gcc (GNU) version 8.3.0
  C compiler flags      :
  Architecture          :
  Using MPI
  Using C files
  Using LibXC
  Using SPGlib
  Using i-PI
  Linking against: /home/knoop/local/anaconda3/pkgs/mkl-2019.4-243/lib/libmkl_intel_lp64.so
                   /home/knoop/local/anaconda3/pkgs/mkl-2019.4-243/lib/libmkl_sequential.so
                   /home/knoop/local/anaconda3/pkgs/mkl-2019.4-243/lib/libmkl_core.so

  #===============================================================================
  # FHI-aims file: ./control.in
  # Created using the Atomic Simulation Environment (ASE)
  # Wed Mar 11 20:43:14 2020
  #===============================================================================
  xc                                 pbesol
  sc_accuracy_rho                    0.001
  compute_analytical_stress          .true.
  relativistic                       atomic_zora scalar
  output_level                       MD_light
  k_grid                             4 4 4
  use_pimd_wrapper                   localhost 12345
  #===============================================================================

  ################################################################################
  #
  #  FHI-aims code project
  #  VB, Fritz-Haber Institut, 2009
  #
  #  Suggested "light" defaults for Mg atom (to be pasted into control.in file)
  #  Be sure to double-check any results obtained with these settings for post-processing,
  #  e.g., with the "tight" defaults and larger basis sets.
  #
  ################################################################################
    species        Mg
  #     global species definitions
      nucleus             12
      mass                24.3050
  #
      l_hartree           4
  #
      cut_pot             4.0          1.5  1.0
      basis_dep_cutoff    1e-4
  #
      radial_base         40 5.5
      radial_multiplier   1
      angular_grids       specified
        division   0.7029   50
        division   0.9689  110
        division   1.1879  194
        division   1.3129  302
  #      division   1.4867  434
  #      division   1.6018  590
  #      division   1.8611  770
  #      division   1.9576  974
  #      division   2.2261 1202
  #      outer_grid   974
        outer_grid   302
  ################################################################################
  #
  #  Definition of "minimal" basis
  #
  ################################################################################
  #     valence basis states
      valence      3  s   2.
      valence      2  p   6.
  #     ion occupancy
      ion_occ      2  s   2.
      ion_occ      2  p   6.
  ################################################################################
  #
  #  Suggested additional basis functions. For production calculations,
  #  uncomment them one after another (the most important basis functions are
  #  listed first).
  #
  #  Constructed for dimers: 2.125 A, 2.375 A, 2.875 A, 3.375 A, 4.5 A
  #
  ################################################################################
  #  "First tier" - improvements: -230.76 meV to -21.94 meV
       hydro 2 p 1.5
       ionic 3 d auto
       hydro 3 s 2.4
  #  "Second tier" - improvements: -5.43 meV to -1.64 meV
  #     hydro 4 f 4.3
  #     hydro 2 p 3.4
  #     hydro 4 s 11.2
  #     hydro 3 d 6.2
  #  "Third tier" - improvements: -0.92 meV to -0.22 meV
  #     hydro 2 s 0.6
  #     hydro 3 p 4.8
  #     hydro 4 f 7.4
  #     hydro 5 g 6.6
  #     hydro 2 p 1.6
  #     hydro 3 d 1.8
  #  "Fourth tier" - improvements: -0.09 meV to -0.05 meV
  #     hydro 4 p 0.45
  #     hydro 5 g 10.4
  #     hydro 2 s 12.4
  #     hydro 4 d 1.7
  ################################################################################
  #
  #  FHI-aims code project
  #  VB, Fritz-Haber Institut, 2009
  #
  #  Suggested "light" defaults for O atom (to be pasted into control.in file)
  #  Be sure to double-check any results obtained with these settings for post-processing,
  #  e.g., with the "tight" defaults and larger basis sets.
  #
  ################################################################################
    species        O
  #     global species definitions
      nucleus             8
      mass                15.9994
  #
      l_hartree           4
  #
      cut_pot             3.5  1.5  1.0
      basis_dep_cutoff    1e-4
  #
      radial_base         36 5.0
      radial_multiplier   1
       angular_grids specified
        division   0.2659   50
        division   0.4451  110
        division   0.6052  194
        division   0.7543  302
  #      division   0.8014  434
  #      division   0.8507  590
  #      division   0.8762  770
  #      division   0.9023  974
  #      division   1.2339 1202
  #      outer_grid 974
        outer_grid 302
  ################################################################################
  #
  #  Definition of "minimal" basis
  #
  ################################################################################
  #     valence basis states
      valence      2  s   2.
      valence      2  p   4.
  #     ion occupancy
      ion_occ      2  s   1.
      ion_occ      2  p   3.
  ################################################################################
  #
  #  Suggested additional basis functions. For production calculations,
  #  uncomment them one after another (the most important basis functions are
  #  listed first).
  #
  #  Constructed for dimers: 1.0 A, 1.208 A, 1.5 A, 2.0 A, 3.0 A
  #
  ################################################################################
  #  "First tier" - improvements: -699.05 meV to -159.38 meV
       hydro 2 p 1.8
       hydro 3 d 7.6
       hydro 3 s 6.4
  #  "Second tier" - improvements: -49.91 meV to -5.39 meV
  #     hydro 4 f 11.6
  #     hydro 3 p 6.2
  #     hydro 3 d 5.6
  #     hydro 5 g 17.6
  #     hydro 1 s 0.75
  #  "Third tier" - improvements: -2.83 meV to -0.50 meV
  #     ionic 2 p auto
  #     hydro 4 f 10.8
  #     hydro 4 d 4.7
  #     hydro 2 s 6.8
  #  "Fourth tier" - improvements: -0.40 meV to -0.12 meV
  #     hydro 3 p 5
  #     hydro 3 s 3.3
  #     hydro 5 g 15.6
  #     hydro 4 f 17.6
  #     hydro 4 d 14
  # Further basis functions - -0.08 meV and below
  #     hydro 3 s 2.1
  #     hydro 4 d 11.6
  #     hydro 3 p 16
  #     hydro 2 s 17.2

  -----------------------------------------------------------------------
  Completed first pass over input file control.in .
  -----------------------------------------------------------------------


  -----------------------------------------------------------------------
  Parsing geometry.in (first pass over file, find array dimensions only).
  The contents of geometry.in will be repeated verbatim below
  unless switched off by setting 'verbatim_writeout .false.' .
  in the first line of geometry.in .
  -----------------------------------------------------------------------

  #=======================================================
  # FHI-aims file: ./geometry.in
  # Created using the Atomic Simulation Environment (ASE)
  # Wed Mar 11 20:43:14 2020
  #=======================================================
  lattice_vector 0.0000000000000000 2.1131357400000002 2.1131357400000002
  lattice_vector 2.1131357400000002 0.0000000000000000 2.1131357400000002
  lattice_vector 2.1131357400000002 2.1131357400000002 0.0000000000000000
  atom_frac 0.0000000000000000 0.0000000000000000 -0.0000000000000000 Mg
  atom_frac 0.5200000000000000 0.5000000000000000 0.5000000000000000 O

  -----------------------------------------------------------------------
  Completed first pass over input file geometry.in .
  -----------------------------------------------------------------------


  Basic array size parameters:
  | Number of species                 :        2
  | Number of atoms                   :        2
  | Number of lattice vectors         :        3
  | Max. basis fn. angular momentum   :        2
  | Max. atomic/ionic basis occupied n:        3
  | Max. number of basis fn. types    :        3
  | Max. radial fns per species/type  :        4
  | Max. logarithmic grid size        :     1334
  | Max. radial integration grid size :       40
  | Max. angular integration grid size:      302
  | Max. angular grid division number :        8
  | Radial grid for Hartree potential :     1334
  | Number of spin channels           :        1

------------------------------------------------------------
          Reading file control.in.
------------------------------------------------------------
  XC: Using PBEsol gradient-corrected functionals.
  Convergence accuracy of self-consistent charge density:  0.1000E-02
  Scalar relativistic treatment of kinetic energy: on-site free-atom approximation to ZORA.
  Requested output level: MD_light
  Found k-point grid:         4         4         4
  Using external wrapper (i-PI) for performing (path integral) molecular dynamics
  **Attention: initial geometry.in file will be ignored!

  Reading configuration options for species Mg                  .
  | Found nuclear charge :  12.0000
  | Found atomic mass :    24.3050000000000      amu
  | Found l_max for Hartree potential  :   4
  | Found cutoff potl. onset [A], width [A], scale factor :    4.00000    1.50000    1.00000
  | Threshold for basis-dependent cutoff potential is   0.100000E-03
  | Found data for basic radial integration grid :    40 points, outermost radius =    5.500 A
  | Found multiplier for basic radial grid :   1
  | Found angular grid specification: user-specified.
  | Specified grid contains     5 separate shells.
  | Check grid settings after all constraints further below.
  | Found free-atom valence shell :  3 s   2.000
  | Found free-atom valence shell :  2 p   6.000
  | Found free-ion valence shell :  2 s   2.000
  | Found free-ion valence shell :  2 p   6.000
  | Found hydrogenic basis function :  2 p   1.500
  | Found ionic basis function :  3 d , default cutoff radius.
  | Found hydrogenic basis function :  3 s   2.400
  Species Mg                  : Missing cutoff potential type.
  Defaulting to exp(1/x)/(1-x)^2 type cutoff potential.
  Species Mg: No 'logarithmic' tag. Using default grid for free atom:
  | Default logarithmic grid data [bohr] : 0.1000E-03 0.1000E+03 0.1012E+01
  | Will include ionic basis functions of  2.0-fold positive Mg                   ion.
  Species Mg: On-site basis accuracy parameter (for Gram-Schmidt orthonormalisation) not specified.
  Using default value basis_acc =  0.1000000E-03.
  Species Mg                  : Using default innermost maximum threshold i_radial=  2 for radial functions.
  Species Mg                  : Default cutoff onset for free atom density etc. : 0.40000000E+01 AA.
  Species Mg                  : Basic radial grid will be enhanced according to radial_multiplier =   1, to contain    40 grid points.

  Reading configuration options for species O                   .
  | Found nuclear charge :   8.0000
  | Found atomic mass :    15.9994000000000      amu
  | Found l_max for Hartree potential  :   4
  | Found cutoff potl. onset [A], width [A], scale factor :    3.50000    1.50000    1.00000
  | Threshold for basis-dependent cutoff potential is   0.100000E-03
  | Found data for basic radial integration grid :    36 points, outermost radius =    5.000 A
  | Found multiplier for basic radial grid :   1
  | Found angular grid specification: user-specified.
  | Specified grid contains     5 separate shells.
  | Check grid settings after all constraints further below.
  | Found free-atom valence shell :  2 s   2.000
  | Found free-atom valence shell :  2 p   4.000
  | Found free-ion valence shell :  2 s   1.000
  | Found free-ion valence shell :  2 p   3.000
  | Found hydrogenic basis function :  2 p   1.800
  | Found hydrogenic basis function :  3 d   7.600
  | Found hydrogenic basis function :  3 s   6.400
  Species O                   : Missing cutoff potential type.
  Defaulting to exp(1/x)/(1-x)^2 type cutoff potential.
  Species O : No 'logarithmic' tag. Using default grid for free atom:
  | Default logarithmic grid data [bohr] : 0.1000E-03 0.1000E+03 0.1012E+01
  Species O : On-site basis accuracy parameter (for Gram-Schmidt orthonormalisation) not specified.
  Using default value basis_acc =  0.1000000E-03.
  Species O                   : Using default innermost maximum threshold i_radial=  2 for radial functions.
  Species O                   : Default cutoff onset for free atom density etc. : 0.35000000E+01 AA.
  Species O                   : Basic radial grid will be enhanced according to radial_multiplier =   1, to contain    36 grid points.

  Finished reading input file 'control.in'. Consistency checks are next.

  MPI_IN_PLACE appears to work with this MPI implementation.
  | Keeping use_mpi_in_place .true. (see manual).
  Target number of points in a grid batch is not set. Defaulting to  100
  Method for grid partitioning is not set. Defaulting to parallel hash+maxmin partitioning.
  Batch size limit is not set. Defaulting to    200
  By default, will store active basis functions for each batch.
  If in need of memory, prune_basis_once .false. can be used to disable this option.
  communication_type for Hartree potential was not specified.
  Defaulting to calc_hartree .
  Defaulting to Pulay charge density mixer.
  Pulay mixer: Number of relevant iterations not set.
  Defaulting to    8 iterations.
  Pulay mixer: Number of initial linear mixing iterations not set.
  Defaulting to    0 iterations.
  Work space size for distributed Hartree potential not set.
  Defaulting to   0.200000E+03 MB.
  Mixing parameter for charge density mixing has not been set.
  Using default: charge_mix_param =     0.0500.
  The mixing parameter will be adjusted in iteration number     2 of the first full s.c.f. cycle only.
  Algorithm-dependent basis array size parameters:
  | n_max_pulay                         :        8
  Maximum number of self-consistency iterations not provided.
  Presetting  1000 iterations.
  Presetting      1001 iterations before the initial mixing cycle
  is restarted anyway using the sc_init_iter criterion / keyword.
  Presetting a factor      1.000 between actual scf density residual
  and density convergence criterion sc_accuracy_rho below which sc_init_iter
  takes no effect.
  Calculation of forces was not defined in control.in. No forces will be calculated.
  Geometry relaxation not requested: no relaxation will be performed.
  Analytical stress will be computed.
  Analytical stress calculation: Only the upper triangle is calculated.
                                 Final output is symmetrized.
  Analytical stress calculation: scf convergence accuracy of stress not set.
                                 Analytical stress self-consistency will not be checked explicitly.
                                 Be sure to set other criteria like sc_accuracy_rho tight enough.
  Force calculation: scf convergence accuracy of forces not set.
  Defaulting to 'sc_accuracy_forces not_checked'.
  Handling of forces: Unphysical translation and rotation will be removed from forces.
  No accuracy limit for integral partition fn. given. Defaulting to  0.1000E-14.
  No threshold value for u(r) in integrations given. Defaulting to  0.1000E-05.
  No occupation type (smearing scheme) given. Defaulting to Gaussian broadening, width =  0.1000E-01 eV.
  The width will be adjusted in iteration number     2 of the first full s.c.f. cycle only.
  S.C.F. convergence parameters will be adjusted in iteration number     2 of the first full s.c.f. cycle only.
  No accuracy for occupation numbers given. Defaulting to  0.1000E-12.
  No threshold value for occupation numbers given. Defaulting to  0.0000E+00.
  No accuracy for fermi level given. Defaulting to  0.1000E-19.
  Maximum # of iterations to find E_F not set. Defaulting to  200.
  Preferred method for the eigenvalue solver ('KS_method') not specified in 'control.in'.
  Defaulting to serial version, LAPACK (via ELSI), since more k-points than CPUs available.
  Will not use alltoall communication since running on < 1024 CPUs.
  Threshold for basis singularities not set.
  Default threshold for basis singularities:  0.1000E-04
  partition_type (choice of integration weights) for integrals was not specified.
  | Using a version of the partition function of Stratmann and coworkers ('stratmann_sparse').
  | At each grid point, the set of atoms used to build the partition table is smoothly restricted to
  | only those atoms whose free-atom density would be non-zero at that grid point.
  Partitioning for Hartree potential was not defined. Using partition_type for integrals.
  | Adjusted default value of keyword multip_moments_threshold to:       0.10000000E-11
  | This value may affect high angular momentum components of the Hartree potential in periodic systems.
  Spin handling was not defined in control.in. Defaulting to unpolarized case.
  Angular momentum expansion for Kerker preconditioner not set explicitly.
  | Using default value of   0
  No explicit requirement for turning off preconditioner.
  | By default, it will be turned off when the charge convergence reaches
  | sc_accuracy_rho  =   0.100000E-02
  No special mixing parameter while Kerker preconditioner is on.
  Using default: charge_mix_param =     0.0500.
  No q(lm)/r^(l+1) cutoff set for long-range Hartree potential.
  | Using default value of  0.100000E-09 .
  | Verify using the multipole_threshold keyword.
  Defaulting to new monopole extrapolation.
  Density update method: automatic selection selected.
  Using density matrix based charge density update.
  Using density matrix based charge density update.
  Using packed matrix style: index .
  Defaulting to use time-reversal symmetry for k-point grid.
------------------------------------------------------------


------------------------------------------------------------
          Reading geometry description geometry.in.
------------------------------------------------------------
  Input structure read successfully.
  The structure contains        2 atoms,  and a total of         20.000 electrons.

  Input geometry:
  | Unit cell:
  |        0.00000000        2.11313574        2.11313574
  |        2.11313574        0.00000000        2.11313574
  |        2.11313574        2.11313574        0.00000000
  | Atomic structure:
  |       Atom                x [A]            y [A]            z [A]
  |    1: Species Mg            0.00000000        0.00000000        0.00000000
  |    2: Species O             2.11313574        2.15539845        2.15539845

  Lattice parameters for 3D lattice (in Angstroms) :     2.988425    2.988425    2.988425
  Angle(s) between unit vectors (in degrees)       :    60.000000   60.000000   60.000000

  | The smallest distance between any two atoms is         2.07130423 AA.
  | The first atom of this pair is atom number                      2 .
  | The second atom of this pair is atom number                     1 .
  | Wigner-Seitz cell of the first atom image           0     1     0 .
  | (The Wigner-Seitz cell of the second atom is 0 0 0  by definition.)

  Symmetry information by spglib:
  | Precision set to  0.1E-04
  | Number of Operations  : 4
  | Space group           : 44
  | International         : Imm2
  | Schoenflies           : C2v^20

  Quantities derived from the lattice vectors:
  | Reciprocal lattice vector 1: -1.486697  1.486697  1.486697
  | Reciprocal lattice vector 2:  1.486697 -1.486697  1.486697
  | Reciprocal lattice vector 3:  1.486697  1.486697 -1.486697
  | Unit cell volume                               :   0.188718E+02  A^3

  Range separation radius for Ewald summation (hartree_convergence_parameter):      2.90574234 bohr.

  Number of empty states per atom not set in control.in - providing a guess from actual geometry.
  | Total number of empty states used during s.c.f. cycle:        6
  If you use a very high smearing, use empty_states (per atom!) in control.in to increase this value.

  Structure-dependent array size parameters:
  | Maximum number of distinct radial functions  :       13
  | Maximum number of basis functions            :       29
  | Number of Kohn-Sham states (occupied + empty):       16
------------------------------------------------------------

************************** W A R N I N G *******************************
* You are using the PIMD wrapper. Specifications and positions         *
* in geometry.in will be IGNORED - all is received from the wrapper.   *
* Please make sure species are declared in the same order in           *
* geometry.in and wrapper input.                                       *
************************************************************************


------------------------------------------------------------
          Preparing all fixed parts of the calculation.
------------------------------------------------------------
  Determining machine precision:
    2.225073858507201E-308
  Setting up grids for atomic and cluster calculations.

  Creating wave function, potential, and density for free atoms.

  Species: Mg

  List of occupied orbitals and eigenvalues:
    n    l              occ      energy [Ha]    energy [eV]
    1    0           2.0000       -46.359170     -1261.4972
    2    0           2.0000        -2.927498       -79.6613
    3    0           2.0000        -0.167982        -4.5710
    2    1           6.0000        -1.706030       -46.4234


  Species: O

  List of occupied orbitals and eigenvalues:
    n    l              occ      energy [Ha]    energy [eV]
    1    0           2.0000       -18.926989      -515.0296
    2    0           2.0000        -0.880247       -23.9527
    2    1           4.0000        -0.331514        -9.0210

  Creating fixed part of basis set: Ionic, confined, hydrogenic.

  Mg                   ion:

  List of free ionic orbitals and eigenvalues:
    n    l      energy [Ha]    energy [eV]
    1    0       -47.122987     -1282.2817
    2    0        -3.674099       -99.9773
    2    1        -2.451441       -66.7071


  List of ionic basis orbitals and eigenvalues:
    n    l      energy [Ha]    energy [eV]    outer radius [A]
    3    2        -0.253797        -6.9062       5.100062


  Mg                   hydrogenic:

  List of hydrogenic basis orbitals:
    n    l      effective z      eigenvalue [eV]  inner max. [A]     outer max. [A]     outer radius [A]
    2    1         1.500000        -7.5955           1.395712           1.395712           5.100062
    3    0         2.400000        -8.0111           0.162319           2.734035           5.100062


  O                    hydrogenic:

  List of hydrogenic basis orbitals:
    n    l      effective z      eigenvalue [eV]  inner max. [A]     outer max. [A]     outer radius [A]
    2    1         1.800000       -10.9749           1.164242           1.164242           4.578029
    3    2         7.600000       -87.3180           0.624125           0.624125           3.251020
    3    0         6.400000       -61.9207           0.061167           1.081902           4.001998


  Adding cutoff potential to free-atom effective potential.
  Creating atomic-like basis functions for current effective potential.

  Species Mg                  :

  List of atomic basis orbitals and eigenvalues:
    n    l      energy [Ha]    energy [eV]    outer radius [A]
    1    0       -46.359170     -1261.4972       0.921046
    2    0        -2.927498       -79.6613       3.621735
    3    0        -0.167982        -4.5710       5.100062
    2    1        -1.706030       -46.4234       4.404175


  Species O                   :

  List of atomic basis orbitals and eigenvalues:
    n    l      energy [Ha]    energy [eV]    outer radius [A]
    1    0       -18.926989      -515.0296       1.415765
    2    0        -0.880247       -23.9527       4.413171
    2    1        -0.331514        -9.0210       4.522403

  Assembling full basis from fixed parts.
  | Species Mg :   atomic orbital   1 s accepted.
  | Species Mg :   atomic orbital   2 s accepted.
  | Species Mg :   atomic orbital   3 s accepted.
  | Species Mg :    hydro orbital   3 s accepted.
  | Species Mg :   atomic orbital   2 p accepted.
  | Species Mg :    hydro orbital   2 p accepted.
  | Species Mg :    ionic orbital   3 d accepted.
  | Species O :   atomic orbital   1 s accepted.
  | Species O :    hydro orbital   3 s accepted.
  | Species O :   atomic orbital   2 s accepted.
  | Species O :   atomic orbital   2 p accepted.
  | Species O :    hydro orbital   2 p accepted.
  | Species O :    hydro orbital   3 d accepted.

  Basis size parameters after reduction:
  | Total number of radial functions:       13
  | Total number of basis functions :       29

  Per-task memory consumption for arrays in subroutine allocate_ext:
  |           4.441052MB.
  Testing on-site integration grid accuracy.
  |  Species  Function  <phi|h_atom|phi> (log., in eV)  <phi|h_atom|phi> (rad., in eV)
           1        1              -1261.4972086064              -1261.4966102663
           1        2                -79.6612686792                -79.6612661850
           1        3                 -4.5766449847                 -4.5764463354
           1        4                  4.2856803862                  4.2813226631
           1        5                -46.4234362291                -46.4234361907
           1        6                 -0.5282558506                 -0.5281575826
           1        7                  4.0144889854                  4.0133736295
           2        8               -515.0295626295               -515.0294562738
           2        9                 15.1698434419                 15.1699322204
           2       10                -21.6038822678                -21.6039118105
           2       11                 -9.0211123393                 -9.0212703513
           2       12                  8.3047696391                  8.2854840999
           2       13                 45.8428042125                 45.8427461222

  Preparing densities etc. for the partition functions (integrals / Hartree potential).

  Preparations completed.
  max(cpu_time)          :      0.155 s.
  Wall clock time (cpu1) :      0.155 s.
------------------------------------------------------------

************************** W A R N I N G *******************************
* Skipping the SCF initialization for now - done inside wrapper      *
************************************************************************

  @ DRIVER MODE: Connecting to host:port localhost       12345
  @ DRIVER MODE: Message from server: STATUS
  @ DRIVER MODE: Message from server: POSDATA
  @ DRIVER MODE: Received positions

------------------------------------------------------------
          Begin self-consistency loop: Initialization.

          Date     :  20200311, Time     :  204314.465
------------------------------------------------------------

  Initializing index lists of integration centers etc. from given atomic structure:
  Mapping all atomic coordinates to central unit cell.

  Initializing the k-points
          Using symmetry for reducing the k-points
  | k-points reduced from:       64 to       36
  | Number of k-points                             :        36
  The eigenvectors in the calculations are COMPLEX.
  | K-points in task   0:         9
  | K-points in task   1:         9
  | K-points in task   2:         9
  | K-points in task   3:         9
  | Number of basis functions in the Hamiltonian integrals :      1946
  | Number of basis functions in a single unit cell        :        29
  | Number of centers in hartree potential         :      1017
  | Number of centers in hartree multipole         :       776
  | Number of centers in electron density summation:       530
  | Number of centers in basis integrals           :       582
  | Number of centers in integrals                 :       189
  | Number of centers in hamiltonian               :       580
  | Consuming        186 KiB for k_phase.
  | Number of super-cells (origin) [n_cells]                     :        2197
  | Number of super-cells (after PM_index) [n_cells]             :         332
  | Number of super-cells in hamiltonian [n_cells_in_hamiltonian]:         332
  | Size of matrix packed + index [n_hamiltonian_matrix_size] :       70918
  | Estimated reciprocal-space cutoff momentum G_max:         3.25739508 bohr^-1 .
  | Reciprocal lattice points for long-range Hartree potential:      64
  Partitioning the integration grid into batches with parallel hashing+maxmin method.
  | Number of batches:      191
  | Maximal batch size:     112
  | Minimal batch size:      54
  | Average batch size:      62.492
  | Standard deviation of batch sizes:       5.898

  Integration load balanced across     4 MPI tasks.
  Work distribution over tasks is as follows:
  Task     0 has       3012 integration points.
  Task     1 has       2986 integration points.
  Task     2 has       2985 integration points.
  Task     3 has       2953 integration points.
  Initializing partition tables, free-atom densities, potentials, etc. across the integration grid (initialize_grid_storage).
  | initialize_grid_storage: Actual outermost partition radius vs. multipole_radius_free
  | (-- VB: in principle, multipole_radius_free should be larger, hence this output)
  | Species        1: Confinement radius =              5.500000000000000 AA, multipole_radius_free =              5.555717568450569 AA.
  | Species        1: outer_partition_radius set to              5.555717568450569 AA .
  | Species        2: Confinement radius =              4.999999999999999 AA, multipole_radius_free =              5.048384829883283 AA.
  | Species        2: outer_partition_radius set to              5.048384829883283 AA .
  | The sparse table of interatomic distances needs       1970.16 kbyte instead of      2709.79 kbyte of memory.
  | Net number of integration points:    11936
  | of which are non-zero points    :     9118
  | Numerical average free-atom electrostatic potential    :    -17.34765164 eV
  Renormalizing the initial density to the exact electron count on the 3D integration grid.
  | Initial density: Formal number of electrons (from input files) :      20.0000000000
  | Integrated number of electrons on 3D grid     :      20.0026739480
  | Charge integration error                      :       0.0026739480
  | Normalization factor for density and gradient :       0.9998663205
  Obtaining max. number of non-zero basis functions in each batch (get_n_compute_maxes).
  | Maximal number of non-zero basis functions:      807 in task     0
  | Maximal number of non-zero basis functions:      785 in task     1
  | Maximal number of non-zero basis functions:      825 in task     2
  | Maximal number of non-zero basis functions:      821 in task     3
  Allocating        0.064 MB for KS_eigenvector_complex
  Integrating Hamiltonian matrix: batch-based integration.
  Time summed over all CPUs for integration: real work        1.226 s, elapsed        1.244 s
  Integrating overlap matrix.
  Time summed over all CPUs for integration: real work        0.906 s, elapsed        0.920 s
  Decreasing sparse matrix size:
   Tolerance:  9.999999824516700E-014
   Hamiltonian matrix
  | Array has    57094 nonzero elements out of    70918 elements
  | Sparsity factor is 0.195
   Overlap matrix
  | Array has    53782 nonzero elements out of    70918 elements
  | Sparsity factor is 0.242
  New size of hamiltonian matrix:       57179

  Updating Kohn-Sham eigenvalues and eigenvectors using ELSI and the (modified) LAPACK eigensolver.
  Singularity check in k-point 4, task 0 (analysis for other k-points/tasks may follow below):

  Obtaining occupation numbers and chemical potential using ELSI.
  | Chemical potential (Fermi level):    -9.39880369 eV
  Writing Kohn-Sham eigenvalues.
  K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]
      1       2.00000         -46.246119        -1258.42094
      2       2.00000         -18.949515         -515.64253
      3       2.00000          -2.939241          -79.98081
      4       2.00000          -1.724501          -46.92607
      5       2.00000          -1.724472          -46.92526
      6       2.00000          -1.724438          -46.92436
      7       2.00000          -1.074630          -29.24216
      8       2.00000          -0.457085          -12.43791
      9       2.00000          -0.456823          -12.43079
     10       2.00000          -0.456820          -12.43071
     11       0.00000          -0.221801           -6.03551
     12       0.00000           0.221833            6.03638
     13       0.00000           0.222549            6.05588
     14       0.00000           0.222984            6.06771
     15       0.00000           0.425167           11.56937
     16       0.00000           0.425678           11.58330

  What follows are estimated values for band gap, HOMO, LUMO, etc.
  | They are estimated on a discrete k-point grid and not necessarily exact.
  | For converged numbers, create a DOS and/or band structure plot on a denser k-grid.

  Highest occupied state (VBM) at    -12.43071357 eV (relative to internal zero)
  | Occupation number:      2.00000000
  | K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  Lowest unoccupied state (CBM) at    -6.03551047 eV (relative to internal zero)
  | Occupation number:      0.00000000
  | K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  ESTIMATED overall HOMO-LUMO gap:      6.39520310 eV between HOMO at k-point 1 and LUMO at k-point 1
  | This appears to be a direct band gap.
  The gap value is above 0.2 eV. Unless you are using a very sparse k-point grid,
  this system is most likely an insulator or a semiconductor.
  Calculating total energy contributions from superposition of free atom densities.

  Total energy components:
  | Sum of eigenvalues            :        -151.77161455 Ha       -4129.91575979 eV
  | XC energy correction          :         -24.34440425 Ha        -662.44494434 eV
  | XC potential correction       :          31.70940006 Ha         862.85667742 eV
  | Free-atom electrostatic energy:        -130.88597416 Ha       -3561.58856859 eV
  | Hartree energy correction     :           0.00000000 Ha           0.00000000 eV
  | Entropy correction            :           0.00000000 Ha           0.00000000 eV
  | ---------------------------
  | Total energy                  :        -275.29259290 Ha       -7491.09259530 eV
  | Total energy, T -> 0          :        -275.29259290 Ha       -7491.09259530 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy        :        -275.29259290 Ha       -7491.09259530 eV

  Derived energy quantities:
  | Kinetic energy                :         276.03701172 Ha        7511.34926225 eV
  | Electrostatic energy          :        -526.98520038 Ha      -14339.99691322 eV
  | Energy correction for multipole
  | error in Hartree potential    :           0.00000000 Ha           0.00000000 eV
  | Sum of eigenvalues per atom                           :       -2064.95787990 eV
  | Total energy (T->0) per atom                          :       -3745.54629765 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy per atom                       :       -3745.54629765 eV
  Initialize hartree_potential_storage
  Max. number of atoms included in rho_multipole:            2

  End scf initialization - timings             :  max(cpu_time)    wall_clock(cpu1)
  | Time for scf. initialization               :        0.904 s           0.904 s
  | Boundary condition initialization          :        0.062 s           0.061 s
  | Integration                                :        0.544 s           0.544 s
  | Solution of K.-S. eqns.                    :        0.007 s           0.007 s
  | Grid partitioning                          :        0.016 s           0.016 s
  | Preloading free-atom quantities on grid    :        0.002 s           0.002 s
  | Free-atom superposition energy             :        0.023 s           0.023 s
  | Total energy evaluation                    :        0.000 s           0.000 s

  Partial memory accounting:
  | Current value for overall tracked memory usage:
  |   Minimum:        1.155 MB (on task 0)
  |   Maximum:        1.155 MB (on task 0)
  |   Average:        1.155 MB
  | Peak value for overall tracked memory usage:
  |   Minimum:        8.148 MB (on task 1 after allocating wave)
  |   Maximum:        8.743 MB (on task 2 after allocating wave)
  |   Average:        8.512 MB
  | Largest tracked array allocation so far:
  |   Minimum:        4.701 MB (hamiltonian_shell on task 1)
  |   Maximum:        5.193 MB (hamiltonian_shell on task 2)
  |   Average:        5.001 MB
  Note:  These values currently only include a subset of arrays which are explicitly tracked.
  The "true" memory usage will be greater.
------------------------------------------------------------
Convergence:    q app. |  density  | eigen (eV) | Etot (eV) | forces (eV/A) |       CPU time |     Clock time
  SCF    1 :  0.28E-02 |  0.27E+00 |   0.14E+01 |  0.25E+00 |             . |        1.078 s |        1.078 s
  SCF    2 :  0.42E-02 |  0.24E+00 |   0.14E+02 |  0.11E+01 |             . |        1.076 s |        1.076 s
  SCF    3 :  0.42E-02 |  0.81E-01 |  -0.54E+01 | -0.20E-01 |             . |        1.068 s |        1.068 s
  SCF    4 :  0.45E-02 |  0.66E-01 |  -0.33E+01 |  0.72E-01 |             . |        1.062 s |        1.063 s
  SCF    5 :  0.35E-02 |  0.32E-01 |   0.37E+01 |  0.15E-01 |             . |        1.069 s |        1.069 s
  SCF    6 :  0.34E-02 |  0.56E-02 |   0.32E-01 |  0.38E-03 |             . |        1.072 s |        1.072 s
  SCF    7 :  0.28E-02 |  0.26E-02 |   0.86E-01 | -0.15E-05 |             . |        1.077 s |        1.077 s
  SCF    8 :  0.27E-02 |  0.49E-03 |  -0.15E-01 |  0.48E-05 |             . |        1.133 s |        1.134 s
  SCF    9 :  0.27E-02 |  0.63E-03 |   0.33E-01 | -0.18E-04 |      0.11E+01 |        1.454 s |        1.453 s
  SCF   10 :  0.27E-02
  Analytical stress tensor components [eV]         xx                  yy                  zz                  xy                  xz                  yz
  -----------------------------------------------------------------------------------------------------------------------------------------------------------
    Nuclear Hellmann-Feynman      :    -0.3870932949E+02   -0.3875207926E+02   -0.3875207926E+02    0.4430429868E-10    0.4427216952E-10    0.2146800652E-01
    Multipole Hellmann-Feynman    :    -0.5289450565E+02   -0.5304899474E+02   -0.5304899474E+02    0.2987199479E-10    0.3011368558E-10    0.8812370048E-01
    On-site Multipole corrections :    -0.1380739258E+00   -0.1382119934E+00   -0.1382119933E+00   -0.1672464662E-10   -0.4833712897E-12    0.2520670526E-03
    Strain deriv. of the orbitals :     0.9174321372E+02    0.9186596802E+02    0.9186596802E+02    0.2265077863E-09    0.5002460457E-09   -0.9978486771E-01
  -----------------------------------------------------------------------------------------------------------------------------------------------------------
  Sum of all contributions        :     0.1304653355E-02   -0.7331797879E-01   -0.7331797897E-01    0.2839594332E-09    0.5741485295E-09    0.1005890634E-01

  +-------------------------------------------------------------------+
  |              Analytical stress tensor - Symmetrized               |
  |                  Cartesian components [eV/A**3]                   |
  +-------------------------------------------------------------------+
  |                x                y                z                |
  |                                                                   |
  |  x         0.00006913       0.00000000       0.00000000           |
  |  y         0.00000000      -0.00388507       0.00053301           |
  |  z         0.00000000       0.00053301      -0.00388507           |
  |                                                                   |
  |  Pressure:       0.00256700   [eV/A**3]                           |
  |                                                                   |
  +-------------------------------------------------------------------+

 * Warning: Stress tensor is anisotropic. Be aware that pressure is an isotropic quantity.

 |  0.12E-03 |   0.32E-01 | -0.12E-04 |      0.12E-02 |       14.861 s |       14.863 s

  Total energy components:
  | Sum of eigenvalues            :        -151.37923990 Ha       -4119.23870235 eV
  | XC energy correction          :         -24.44685738 Ha        -665.23283605 eV
  | XC potential correction       :          31.84308529 Ha         866.49443746 eV
  | Free-atom electrostatic energy:        -130.88597416 Ha       -3561.58856859 eV
  | Hartree energy correction     :          -0.37115708 Ha         -10.09969792 eV
  | Entropy correction            :           0.00000000 Ha           0.00000000 eV
  | ---------------------------
  | Total energy                  :        -275.24014324 Ha       -7489.66536745 eV
  | Total energy, T -> 0          :        -275.24014324 Ha       -7489.66536745 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy        :        -275.24014324 Ha       -7489.66536745 eV

  Derived energy quantities:
  | Kinetic energy                :         275.72011310 Ha        7502.72601198 eV
  | Electrostatic energy          :        -526.51339896 Ha      -14327.15854338 eV
  | Energy correction for multipole
  | error in Hartree potential    :           0.00356157 Ha           0.09691537 eV
  | Sum of eigenvalues per atom                           :       -2059.61935117 eV
  | Total energy (T->0) per atom                          :       -3744.83268372 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy per atom                       :       -3744.83268372 eV
  What follows are estimated values for band gap, HOMO, LUMO, etc.
  | They are estimated on a discrete k-point grid and not necessarily exact.
  | For converged numbers, create a DOS and/or band structure plot on a denser k-grid.

  Highest occupied state (VBM) at     -9.64325638 eV (relative to internal zero)
  | Occupation number:      2.00000000
  | K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  Lowest unoccupied state (CBM) at    -5.16542680 eV (relative to internal zero)
  | Occupation number:      0.00000000
  | K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  ESTIMATED overall HOMO-LUMO gap:      4.47782958 eV between HOMO at k-point 1 and LUMO at k-point 1
  | This appears to be a direct band gap.
  The gap value is above 0.2 eV. Unless you are using a very sparse k-point grid,
  this system is most likely an insulator or a semiconductor.

  Self-consistency cycle converged.

  Removing unitary transformations (pure translations, rotations) from forces on atoms.
  Atomic forces before filtering:
  | Net force on center of mass :  -0.379388E-08 -0.426476E-02 -0.426475E-02 eV/A
  Atomic forces after filtering:
  | Net force on center of mass :  -0.265846E-23 -0.446015E-16  0.000000E+00 eV/A

  Energy and forces in a compact form:
  | Total energy uncorrected      :         -0.748966536744980E+04 eV
  | Total energy corrected        :         -0.748966536744980E+04 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy        :         -0.748966536744980E+04 eV
  Total atomic forces (unitary forces cleaned) [eV/Ang]:
  |    1         -0.108555415821635E-07          0.232312241621526E+00          0.232312248124934E+00
  |    2          0.108555415821635E-07         -0.232312241621526E+00         -0.232312248124934E+00

  Removing unitary transformations (pure translations, rotations) from forces on atoms.
  Atomic forces before filtering:
  | Net force on center of mass :  -0.265846E-23 -0.446015E-16  0.000000E+00 eV/A
  Atomic forces after filtering:
  | Net force on center of mass :   0.000000E+00  0.446015E-16  0.000000E+00 eV/A
 ------------------------------------------------------------------------
 Atomic structure that was used in the preceding time step of the wrapper
                         x [A]             y [A]             z [A]
  lattice_vector         0.00000000        2.11313574        2.11313574
  lattice_vector         2.11313574        0.00000000        2.11313574
  lattice_vector         2.11313574        2.11313574        0.00000000

            atom         0.00000000        0.00000000        0.00000000  Mg
            atom         2.11313574        0.04226271        0.04226271  O
 ------------------------------------------------------------------------
  @ DRIVER MODE: Message from server: STATUS
  @ DRIVER MODE: Message from server: GETFORCE
  @ DRIVER MODE: Returning v,forces,stress
  @ DRIVER MODE: Message from server: STATUS
  @ DRIVER MODE: Message from server: INIT
  @ DRIVER MODE: Receiving replica           0
  @ DRIVER MODE: Message from server: STATUS
  @ DRIVER MODE: Message from server: POSDATA
  @ DRIVER MODE: Received positions

------------------------------------------------------------
          Begin self-consistency loop: Re-initialization.

  Date     :  20200311, Time     :  204340.453
------------------------------------------------------------

  Initializing index lists of integration centers etc. from given atomic structure:
  Mapping all atomic coordinates to central unit cell.

  Initializing the k-points
          Using symmetry for reducing the k-points
  | k-points reduced from:       64 to       36
  | Number of k-points                             :        36
  The eigenvectors in the calculations are COMPLEX.
  | K-points in task   0:         9
  | K-points in task   1:         9
  | K-points in task   2:         9
  | K-points in task   3:         9
  | Number of basis functions in the Hamiltonian integrals :      1949
  | Number of basis functions in a single unit cell        :        29
  | Number of centers in hartree potential         :      1097
  | Number of centers in hartree multipole         :       842
  | Number of centers in electron density summation:       585
  | Number of centers in basis integrals           :       633
  | Number of centers in integrals                 :       203
  | Number of centers in hamiltonian               :       654
  | Consuming        228 KiB for k_phase.
  | Number of super-cells (origin) [n_cells]                     :        2197
  | Number of super-cells (after PM_index) [n_cells]             :         406
  | Number of super-cells in hamiltonian [n_cells_in_hamiltonian]:         406
  | Size of matrix packed + index [n_hamiltonian_matrix_size] :       70804
  Partitioning the integration grid into batches with parallel hashing+maxmin method.
  Initializing partition tables, free-atom densities, potentials, etc. across the integration grid (initialize_grid_storage).
  | Species        1: outer_partition_radius set to              5.555717568450569 AA .
  | Species        2: outer_partition_radius set to              5.048384829883283 AA .
  | The sparse table of interatomic distances needs       2198.75 kbyte instead of      3205.51 kbyte of memory.
  | Net number of integration points:    11936
  | of which are non-zero points    :     9174
  Renormalizing the initial density to the exact electron count on the 3D integration grid.
  | Initial density: Formal number of electrons (from input files) :      20.0000000000
  | Integrated number of electrons on 3D grid     :      20.0033111186
  | Charge integration error                      :       0.0033111186
  | Normalization factor for density and gradient :       0.9998344715
  Calculating total energy contributions from superposition of free atom densities.
  Initialize hartree_potential_storage
  Max. number of atoms included in rho_multipole:            2
  Integrating overlap matrix.
  Time summed over all CPUs for integration: real work        1.006 s, elapsed        1.029 s
  Orthonormalizing eigenvectors

  End scf reinitialization - timings           :  max(cpu_time)    wall_clock(cpu1)
  | Time for scf. reinitialization             :        0.696 s           0.697 s
  | Boundary condition initialization          :        0.095 s           0.095 s
  | Integration                                :        0.258 s           0.258 s
  | Grid partitioning                          :        0.019 s           0.019 s
  | Preloading free-atom quantities on grid    :        0.296 s           0.295 s
  | Free-atom superposition energy             :        0.026 s           0.027 s
  | K.-S. eigenvector reorthonormalization     :        0.002 s           0.001 s
------------------------------------------------------------
Convergence:    q app. |  density  | eigen (eV) | Etot (eV) | forces (eV/A) |       CPU time |     Clock time
  SCF    1 :  0.33E-02 |  0.39E+00 |  -0.41E+04 | -0.75E+04 |             . |        1.072 s |        1.072 s
  SCF    2 :  0.32E-02 |  0.91E+00 |  -0.22E+02 |  0.79E+01 |             . |        1.400 s |        1.400 s
  SCF    3 :  0.31E-02 |  0.56E+00 |  -0.39E+02 |  0.45E+01 |             . |        1.449 s |        1.450 s
  SCF    4 :  0.31E-02 |  0.59E-01 |   0.28E+01 |  0.14E-01 |             . |        1.252 s |        1.253 s
  SCF    5 :  0.32E-02 |  0.25E-01 |  -0.32E+00 | -0.21E-02 |             . |        1.350 s |        1.349 s
  SCF    6 :  0.33E-02 |  0.32E-02 |  -0.25E+00 |  0.29E-03 |             . |        1.176 s |        1.175 s
  SCF    7 :  0.35E-02 |  0.49E-02 |   0.60E+00 |  0.44E-04 |             . |        1.185 s |        1.186 s
  SCF    8 :  0.34E-02 |  0.11E-02 |   0.10E+00 | -0.69E-04 |             . |        1.165 s |        1.165 s
  SCF    9 :  0.34E-02 |  0.82E-04 |   0.41E-02 | -0.33E-05 |             . |        1.168 s |        1.168 s
  SCF   10 :  0.33E-02 |  0.72E-04 |  -0.19E-02 | -0.80E-05 |      0.33E-02 |        1.642 s |        1.642 s
  SCF   11 :  0.33E-02
  Analytical stress tensor components [eV]         xx                  yy                  zz                  xy                  xz                  yz
  -----------------------------------------------------------------------------------------------------------------------------------------------------------
    Nuclear Hellmann-Feynman      :    -0.3864674379E+02   -0.3864415535E+02   -0.3864415535E+02   -0.2513001734E-11   -0.2404967012E-11    0.2659262711E-04
    Multipole Hellmann-Feynman    :    -0.5297314950E+02   -0.5297337267E+02   -0.5297337267E+02   -0.2417076525E-11   -0.1885368919E-11    0.4490875098E-05
    On-site Multipole corrections :    -0.1375921434E+00   -0.1375921435E+00   -0.1375921434E+00    0.1740136643E-11    0.5800455476E-11    0.4302004478E-11
    Strain deriv. of the orbitals :     0.9175132185E+02    0.9174867058E+02    0.9174867058E+02   -0.4651096734E-09   -0.3547552527E-09   -0.1085119856E-04
  -----------------------------------------------------------------------------------------------------------------------------------------------------------
  Sum of all contributions        :    -0.6163591118E-02   -0.6449573826E-02   -0.6449574530E-02   -0.4682996150E-09   -0.3532451331E-09    0.2023230795E-04

  +-------------------------------------------------------------------+
  |              Analytical stress tensor - Symmetrized               |
  |                  Cartesian components [eV/A**3]                   |
  +-------------------------------------------------------------------+
  |                x                y                z                |
  |                                                                   |
  |  x        -0.00032660      -0.00000000      -0.00000000           |
  |  y        -0.00000000      -0.00034176       0.00000107           |
  |  z        -0.00000000       0.00000107      -0.00034176           |
  |                                                                   |
  |  Pressure:       0.00033671   [eV/A**3]                           |
  |                                                                   |
  +-------------------------------------------------------------------+

 * Warning: Stress tensor is anisotropic. Be aware that pressure is an isotropic quantity.

 |  0.17E-04 |   0.99E-03 | -0.18E-05 |      0.29E-02 |       16.571 s |       16.572 s

  Total energy components:
  | Sum of eigenvalues            :        -151.38002476 Ha       -4119.26005928 eV
  | XC energy correction          :         -24.44607870 Ha        -665.21164709 eV
  | XC potential correction       :          31.84204127 Ha         866.46602823 eV
  | Free-atom electrostatic energy:        -130.88928774 Ha       -3561.67873571 eV
  | Hartree energy correction     :          -0.36705358 Ha          -9.98803622 eV
  | Entropy correction            :           0.00000000 Ha           0.00000000 eV
  | ---------------------------
  | Total energy                  :        -275.24040352 Ha       -7489.67245007 eV
  | Total energy, T -> 0          :        -275.24040352 Ha       -7489.67245007 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy        :        -275.24040352 Ha       -7489.67245007 eV

  Derived energy quantities:
  | Kinetic energy                :         275.71490212 Ha        7502.58421404 eV
  | Electrostatic energy          :        -526.50922694 Ha      -14327.04501702 eV
  | Energy correction for multipole
  | error in Hartree potential    :           0.00349067 Ha           0.09498603 eV
  | Sum of eigenvalues per atom                           :       -2059.63002964 eV
  | Total energy (T->0) per atom                          :       -3744.83622503 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy per atom                       :       -3744.83622503 eV
  What follows are estimated values for band gap, HOMO, LUMO, etc.
  | They are estimated on a discrete k-point grid and not necessarily exact.
  | For converged numbers, create a DOS and/or band structure plot on a denser k-grid.

  Highest occupied state (VBM) at     -9.63993423 eV (relative to internal zero)
  | Occupation number:      2.00000000
  | K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  Lowest unoccupied state (CBM) at    -5.16099556 eV (relative to internal zero)
  | Occupation number:      0.00000000
  | K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)

  ESTIMATED overall HOMO-LUMO gap:      4.47893868 eV between HOMO at k-point 1 and LUMO at k-point 1
  | This appears to be a direct band gap.
  The gap value is above 0.2 eV. Unless you are using a very sparse k-point grid,
  this system is most likely an insulator or a semiconductor.

  Self-consistency cycle converged.

  Removing unitary transformations (pure translations, rotations) from forces on atoms.
  Atomic forces before filtering:
  | Net force on center of mass :  -0.173882E-08  0.213870E-03  0.213872E-03 eV/A
  Atomic forces after filtering:
  | Net force on center of mass :  -0.132923E-23  0.000000E+00  0.653343E-19 eV/A

  Energy and forces in a compact form:
  | Total energy uncorrected      :         -0.748967245006648E+04 eV
  | Total energy corrected        :         -0.748967245006648E+04 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy        :         -0.748967245006648E+04 eV
  Total atomic forces (unitary forces cleaned) [eV/Ang]:
  |    1          0.773348206613320E-08         -0.167235616064691E-03         -0.167231728110871E-03
  |    2         -0.773348206613320E-08          0.167235616064691E-03          0.167231728110871E-03

  Removing unitary transformations (pure translations, rotations) from forces on atoms.
  Atomic forces before filtering:
  | Net force on center of mass :  -0.132923E-23  0.000000E+00  0.653343E-19 eV/A
  Atomic forces after filtering:
  | Net force on center of mass :  -0.132923E-23  0.000000E+00  0.217781E-19 eV/A
 ------------------------------------------------------------------------
 Atomic structure that was used in the preceding time step of the wrapper
                         x [A]             y [A]             z [A]
  lattice_vector         0.00000000        2.11313574        2.11313574
  lattice_vector         2.11313574        0.00000000        2.11313574
  lattice_vector         2.11313574        2.11313574        0.00000000

            atom         0.00000000        0.00000000        0.00000000  Mg
            atom         2.11313574        2.11313574        2.11313574  O
 ------------------------------------------------------------------------
  @ DRIVER MODE: Message from server: STATUS
  @ DRIVER MODE: Message from server: GETFORCE
  @ DRIVER MODE: Returning v,forces,stress

------------------------------------------------------------------------------
  Final output of selected total energy values:

  The following output summarizes some interesting total energy values
  at the end of a run (AFTER all relaxation, molecular dynamics, etc.).

  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :          -7489.672450066 eV
  | Final zero-broadening corrected energy (caution - metals only) :          -7489.672450066 eV
  | For reference only, the value of 1 Hartree used in FHI-aims is :             27.211384500 eV
  | For reference only, the overall average (free atom contribution
  | + realspace contribution) of the electrostatic potential after
  | s.c.f. convergence is                                          :            -16.786786172 eV

  Before relying on these values, please be sure to understand exactly which
  total energy value is referred to by a given number. Different objects may
  all carry the same name 'total energy'. Definitions:

  Total energy of the DFT / Hartree-Fock s.c.f. calculation:
  | Note that this energy does not include ANY quantities calculated after the
  | s.c.f. cycle, in particular not ANY RPA, MP2, etc. many-body perturbation terms.

  Final zero-broadening corrected energy:
  | For metallic systems only, a broadening of the occupation numbers at the Fermi
  | level can be extrapolated back to zero broadening by an electron-gas inspired
  | formula. For all systems that are not real metals, this value can be
  | meaningless and should be avoided.

------------------------------------------------------------------------------
  Methods described in the following list of references were used in this FHI-aims run.
  If you publish the results, please make sure to cite these reference if they apply.
  FHI-aims is an academic code, and for our developers (often, Ph.D. students
  and postdocs), scientific credit in the community is essential.
  Thank you for helping us!

  For any use of FHI-aims, please cite:

    Volker Blum, Ralf Gehrke, Felix Hanke, Paula Havu, Ville Havu,
    Xinguo Ren, Karsten Reuter, and Matthias Scheffler
    'Ab initio molecular simulations with numeric atom-centered orbitals'
    Computer Physics Communications 180, 2175-2196 (2009)
    http://dx.doi.org/10.1016/j.cpc.2009.06.022


  For the analytical stress tensor used in your run, please cite:

    Franz Knuth, Christian Carbogno, Viktor Atalla, Volker Blum, Matthias Scheffler
    'All-electron formalism for total energy strain derivatives and
    stress tensor components for numeric atom-centered orbitals'
    Computer Physics Communications 190, 33-50 (2015).
    http://dx.doi.org/10.1016/j.cpc.2015.01.003


  The provided symmetry information was generated with SPGlib:

    Atsushi Togo, Yusuke Seto, Dimitar Pashov
    SPGlib 1.7.3 obtained from http://spglib.sourceforge.net
    Copyright (C) 2008 Atsushi Togo


  The ELSI infrastructure was used in your run to solve the Kohn-Sham electronic structure.
  Please check out http://elsi-interchange.org to learn more.
  If scalability is important for your project, please acknowledge ELSI by citing:

    V. W-z. Yu, F. Corsetti, A. Garcia, W. P. Huhn, M. Jacquelin, W. Jia,
    B. Lange, L. Lin, J. Lu, W. Mi, A. Seifitokaldani, A. Vazquez-Mayagoitia,
    C. Yang, H. Yang, and V. Blum
    'ELSI: A unified software interface for Kohn-Sham electronic structure solvers'
    Computer Physics Communications 222, 267-285 (2018).
    http://dx.doi.org/10.1016/j.cpc.2017.09.007


  For the real-space grid partitioning and parallelization used in this calculation, please cite:

    Ville Havu, Volker Blum, Paula Havu, and Matthias Scheffler,
    'Efficient O(N) integration for all-electron electronic structure calculation'
    'using numerically tabulated basis functions'
    Journal of Computational Physics 228, 8367-8379 (2009).
    http://dx.doi.org/10.1016/j.jcp.2009.08.008

  Of course, there are many other important community references, e.g., those cited in the
  above references. Our list is limited to references that describe implementations in the
  FHI-aims code. The reason is purely practical (length of this list) - please credit others as well.

------------------------------------------------------------
          Leaving FHI-aims.
          Date     :  20200311, Time     :  204410.632

          Computational steps:
          | Number of self-consistency cycles          :           21
          | Number of SCF (re)initializations          :            2

          Detailed time accounting                     :  max(cpu_time)    wall_clock(cpu1)
          | Total time                                 :       56.355 s          56.379 s
          | Preparation time                           :        0.155 s           0.155 s
          | Boundary condition initalization           :        0.157 s           0.156 s
          | Grid partitioning                          :        0.035 s           0.035 s
          | Preloading free-atom quantities on grid    :        0.298 s           0.297 s
          | Free-atom superposition energy             :        0.049 s           0.050 s
          | Total time for integrations                :        8.101 s           8.102 s
          | Total time for solution of K.-S. equations :        0.136 s           0.136 s
          | Total time for density & force components  :       40.500 s          40.501 s
          | Total time for mixing & preconditioning    :        1.391 s           1.394 s
          | Total time for Hartree multipole update    :        0.013 s           0.012 s
          | Total time for Hartree multipole sum       :        5.042 s           5.044 s
          | Total time for total energy evaluation     :        0.002 s           0.001 s
          | Total time for scaled ZORA corrections     :        0.000 s           0.000 s

          Partial memory accounting:
          | Residual value for overall tracked memory usage across tasks:     0.000000 MB (should be 0.000000 MB)
          | Peak values for overall tracked memory usage:
          |   Minimum:       20.033 MB (on task 1 after allocating d_wave)
          |   Maximum:       20.979 MB (on task 0 after allocating d_wave)
          |   Average:       20.447 MB
          | Largest tracked array allocation:
          |   Minimum:        5.130 MB (density_matrix_con on task 1)
          |   Maximum:        5.538 MB (density_matrix_con on task 0)
          |   Average:        5.251 MB
          Note:  These values currently only include a subset of arrays which are explicitly tracked.
          The "true" memory usage will be greater.

          Have a nice day.
------------------------------------------------------------
"""

    with open("aims.out", "w") as f:
        f.write(output)
