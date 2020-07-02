def test_combine():
    import os
    from numpy.random import RandomState

    from ase.build import molecule
    from ase.vibrations import Vibrations
    from ase.vibrations import Infrared


    class RandomCalculator():
        """Fake Calculator class.

        """
        def __init__(self):
            self.rng = RandomState(42)

        def get_forces(self, atoms):
            return self.rng.rand(len(atoms), 3)

        def get_dipole_moment(self, atoms):
            return self.rng.rand(3)


    atoms = molecule('C2H6')
    ir = Infrared(atoms)
    ir.calc = RandomCalculator()
    ir.run()
    freqs = ir.get_frequencies()
    ints = ir.intensities
    assert ir.combine() == 49

    ir = Infrared(atoms)
    assert (freqs == ir.get_frequencies()).all()
    assert (ints == ir.intensities).all()

    vib = Vibrations(atoms, name='ir')
    assert (freqs == vib.get_frequencies()).all()

    # Read the data from other working directory
    dirname = os.path.basename(os.getcwd())
    os.chdir('..')  # Change working directory
    ir = Infrared(atoms, name=os.path.join(dirname, 'ir'))
    assert (freqs == ir.get_frequencies()).all()
    os.chdir(dirname)

    ir = Infrared(atoms)
    assert ir.split() == 1
    assert (freqs == ir.get_frequencies()).all()
    assert (ints == ir.intensities).all()

    vib = Vibrations(atoms, name='ir')
    assert (freqs == vib.get_frequencies()).all()

    assert ir.clean() == 49
