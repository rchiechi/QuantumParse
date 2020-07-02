def test_neb(plt):
    from ase import Atoms
    from ase.constraints import FixAtoms
    import ase.io
    from ase.neb import NEB, NEBTools
    from ase.calculators.morse import MorsePotential
    from ase.optimize import BFGS, QuasiNewton


    def calc():
        # Common calculator for all images.
        return MorsePotential()


    # Create and relax initial and final states.
    initial = Atoms('H7',
                    positions=[(0, 0, 0),
                               (1, 0, 0),
                               (0, 1, 0),
                               (1, 1, 0),
                               (0, 2, 0),
                               (1, 2, 0),
                               (0.5, 0.5, 1)],
                    constraint=[FixAtoms(range(6))],
                    calculator=calc())
    dyn = QuasiNewton(initial)
    dyn.run(fmax=0.01)

    final = initial.copy()
    final.calc = calc()
    final.positions[6, 1] = 2 - initial.positions[6, 1]
    dyn = QuasiNewton(final)
    dyn.run(fmax=0.01)

    # Run NEB without climbing image.
    fmax = 0.05
    nimages = 4

    images = [initial]
    for index in range(nimages - 2):
        images += [initial.copy()]
        images[-1].calc = calc()
    images += [final]

    neb = NEB(images)
    neb.interpolate()

    dyn = BFGS(neb, trajectory='mep.traj')
    dyn.run(fmax=fmax)

    # Check climbing image.
    neb.climb = True
    dyn.run(fmax=fmax)

    # Check NEB tools.
    nt_images = ase.io.read('mep.traj', index='-{:d}:'.format(nimages))
    nebtools = NEBTools(nt_images)
    nt_fmax = nebtools.get_fmax(climb=True)
    Ef, dE = nebtools.get_barrier()
    print(Ef, dE, fmax, nt_fmax)
    assert nt_fmax < fmax
    assert abs(Ef - 1.389) < 0.001
    # Plot one band.
    nebtools.plot_band()
    # Plot many (ok, 2) bands.
    nt_images = ase.io.read('mep.traj', index='-{:d}:'.format(2 * nimages))
    nebtools = NEBTools(nt_images)
    nebtools.plot_bands()
