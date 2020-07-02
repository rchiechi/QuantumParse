def test_readwrite_errors():
    import pytest
    from io import StringIO
    from ase.io import read, write
    from ase.build import bulk
    from ase.io.formats import UnknownFileTypeError

    atoms = bulk('Au')
    fd = StringIO()

    with pytest.raises(UnknownFileTypeError):
        write(fd, atoms, format='hello')

    with pytest.raises(UnknownFileTypeError):
        read(fd, format='hello')
