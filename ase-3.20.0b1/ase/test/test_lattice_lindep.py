def test_lattice_lindep():
    import pytest
    from ase.lattice.cubic import FaceCenteredCubic
    from ase.lattice.hexagonal import HexagonalClosedPacked

    with pytest.raises(ValueError):
        # The Miller indices of the surfaces are linearly dependent
        atoms = FaceCenteredCubic(symbol='Cu',
                                  miller=[[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    # This one should be OK:
    atoms = FaceCenteredCubic(symbol='Cu',
                              miller=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    print(atoms.get_cell())


    with pytest.raises(ValueError):
        # The directions spanning the unit cell are linearly dependent
        atoms = FaceCenteredCubic(symbol='Cu',
                                  directions=[[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    with pytest.raises(ValueError):
        # The directions spanning the unit cell are linearly dependent
        atoms = FaceCenteredCubic(symbol='Cu',
                                  directions=[[1, 1, 0], [1, 0, 0], [0, 1, 0]])

    # This one should be OK:
    atoms = FaceCenteredCubic(symbol='Cu',
                              directions=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    print(atoms.get_cell())

    with pytest.raises((ValueError, NotImplementedError)):
        # The Miller indices of the surfaces are linearly dependent
        atoms = HexagonalClosedPacked(symbol='Mg',
                                      miller=[[1, -1, 0, 0],
                                              [1, 0, -1, 0],
                                              [0, 1, -1, 0]])

    # This one should be OK
    #
    # It is not!  The miller argument is broken in hexagonal crystals!
    #
    # atoms = HexagonalClosedPacked(symbol='Mg',
    #                               miller=[[1, -1, 0, 0],
    #                                       [1, 0, -1, 0],
    #                                       [0, 0, 0, 1]])
    # print(atoms.get_cell())

    with pytest.raises(ValueError):
        # The directions spanning the unit cell are linearly dependent
        atoms = HexagonalClosedPacked(symbol='Mg',
                                      directions=[[1, -1, 0, 0],
                                                  [1, 0, -1, 0],
                                                  [0, 1, -1, 0]])

    # This one should be OK
    atoms = HexagonalClosedPacked(symbol='Mg',
                                  directions=[[1, -1, 0, 0],
                                              [1, 0, -1, 0],
                                              [0, 0, 0, 1]])
    print(atoms.get_cell())
