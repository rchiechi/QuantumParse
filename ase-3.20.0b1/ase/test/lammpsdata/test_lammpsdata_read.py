def test_lammpsdata_read(datadir):
    import ase.io
    from ase.utils import StringIO

    lammps_data_path = datadir / 'lammpsdata_input.data'
    lammps_data = ase.io.read(lammps_data_path, format="lammps-data",
                              units="metal")

    expected_output_path = datadir / 'lammpsdata_expected_output.xyz'
    with open(expected_output_path) as f:
        expected_output = f.read().splitlines()

    buf = StringIO()
    ase.io.write(buf, lammps_data, format="extxyz",
                 columns=["symbols",
                          "positions",
                          "bonds",
                          "masses",
                          "mol-id",
                          "numbers",
                          "angles",
                          "momenta",
                          "type",
                          "id",
                          "travel"],
                 write_info=False)

    lines = [line.strip() for line in buf.getvalue().split("\n")]

    assert lines == expected_output
