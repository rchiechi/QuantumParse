import numpy as np
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.phonons import Phonons
from ase.thermochemistry import (IdealGasThermo, HarmonicThermo,
                                 CrystalThermo, HinderedThermo)
from ase.calculators.emt import EMT


def test_ideal_gas_thermo():
    atoms = Atoms('N2',
                  positions=[(0, 0, 0), (0, 0, 1.1)])
    atoms.calc = EMT()
    QuasiNewton(atoms).run(fmax=0.01)
    energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, name='idealgasthermo-vib')
    vib.run()
    vib_energies = vib.get_energies()

    thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                            atoms=atoms, symmetrynumber=2, spin=0,
                            potentialenergy=energy)
    thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.)

    # Harmonic thermo.

def test_harmonic_thermo():
    atoms = fcc100('Cu', (2, 2, 2), vacuum=10.)
    atoms.calc = EMT()
    add_adsorbate(atoms, 'Pt', 1.5, 'hollow')
    atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms
                                           if atom.symbol == 'Cu']))
    QuasiNewton(atoms).run(fmax=0.01)
    vib = Vibrations(atoms, name='harmonicthermo-vib',
                     indices=[atom.index for atom in atoms
                              if atom.symbol != 'Cu'])
    vib.run()
    vib.summary()
    vib_energies = vib.get_energies()

    thermo = HarmonicThermo(vib_energies=vib_energies,
                            potentialenergy=atoms.get_potential_energy())
    thermo.get_helmholtz_energy(temperature=298.15)


def test_crystal_thermo(asap3):
    atoms = bulk('Al', 'fcc', a=4.05)
    calc = asap3.EMT()
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Phonon calculator
    N = 7
    ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
    ph.run()

    ph.read(acoustic=True)
    phonon_energies, phonon_DOS = ph.dos(kpts=(4, 4, 4), npts=30,
                                         delta=5e-4)

    thermo = CrystalThermo(phonon_energies=phonon_energies,
                           phonon_DOS=phonon_DOS,
                           potentialenergy=energy,
                           formula_units=4)
    thermo.get_helmholtz_energy(temperature=298.15)

    # Hindered translator / rotor.
    # (Taken directly from the example given in the documentation.)

def test_hindered_thermo():
    vibs = np.array([3049.060670,
                     3040.796863,
                     3001.661338,
                     2997.961647,
                     2866.153162,
                     2750.855460,
                     1436.792655,
                     1431.413595,
                     1415.952186,
                     1395.726300,
                     1358.412432,
                     1335.922737,
                     1167.009954,
                     1142.126116,
                     1013.918680,
                     803.400098,
                     783.026031,
                     310.448278,
                     136.112935,
                     112.939853,
                     103.926392,
                     77.262869,
                     60.278004,
                     25.825447])
    vib_energies = vibs / 8065.54429  # Convert to eV from cm^-1.
    trans_barrier_energy = 0.049313  # eV
    rot_barrier_energy = 0.017675  # eV
    sitedensity = 1.5e15  # cm^-2
    rotationalminima = 6
    symmetrynumber = 1
    mass = 30.07  # amu
    inertia = 73.149  # amu Ang^-2

    thermo = HinderedThermo(vib_energies=vib_energies,
                            trans_barrier_energy=trans_barrier_energy,
                            rot_barrier_energy=rot_barrier_energy,
                            sitedensity=sitedensity,
                            rotationalminima=rotationalminima,
                            symmetrynumber=symmetrynumber,
                            mass=mass,
                            inertia=inertia)

    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    target = 1.593  # Taken from documentation example.
    assert (helmholtz - target) < 0.001
