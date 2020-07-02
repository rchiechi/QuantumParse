"""Test reading/writing in ASE on pathlib objects"""

from pathlib import Path
import io
from ase.build import molecule
from ase.io import read, write
from ase.utils import PurePath, convert_string_to_fd, reader, writer


# Test reader/writer
teststr = 'Teststring!'
@writer
def mywrite(file, fdcmp=None):
    assert isinstance(file, io.TextIOBase)
    assert file.mode == 'w'

    # Print something to the file
    print(teststr, file=file)

    # Check that we didn't change the fd
    if fdcmp:
        assert file is fdcmp


@reader
def myread(file, fdcmp=None):
    assert isinstance(file, io.TextIOBase)
    assert file.mode == 'r'

    # Ensure we can read from file
    line = next(file)
    assert line.strip() == teststr

    # Check that we didn't change the fd
    if fdcmp:
        assert file is fdcmp


def test_pathlib_support():
    path = Path('tmp_plib_testdir')

    # Test PurePath catches path
    assert isinstance(path, PurePath)


    path.mkdir(exist_ok=True)

    myf = path / 'test.txt'

    fd = convert_string_to_fd(myf)
    assert isinstance(fd, io.TextIOBase)
    fd.close()

    fd = convert_string_to_fd(str(myf))
    assert isinstance(fd, io.TextIOBase)
    fd.close()


    for f in [myf, str(myf)]:
        myf.unlink()                # Remove the file first
        mywrite(f)
        myread(f)


    # Check reader, writer on open filestream
    # Here, the filestream shouldn't be altered
    with myf.open('w') as f:
        mywrite(f, fdcmp=f)
    with myf.open('r') as f:
        myread(f, fdcmp=f)

    # Check that we can read and write atoms with pathlib
    atoms = molecule('H2', vacuum=5)
    f2 = path / 'test2.txt'
    for form in ['vasp', 'traj', 'xyz']:
        write(f2, atoms, format=form)
        read(f2, format=form)
