def test_gaussian_cmdline(cli):
    from ase.db import connect
    from ase.io import read
    from ase.io.jsonio import read_json

    cli.shell("""\
    ase build O O.xyz && ase run gaussian O.xyz -o gaussian_cmdline.json &&
    ase build O2 O2.xyz && ase run gaussian O2.xyz -o gaussian_cmdline.json""",
        'gaussian')
    c = connect('gaussian_cmdline.json')
    dct = read_json('gaussian_cmdline.json')
    for index, name in enumerate(['O', 'O2']):
        d = c.get(index + 1)
        id = d.id
        e1 = d.energy
        e2 = c.get_atoms(id).get_potential_energy()
        e3 = read(name + '.log').get_potential_energy()
        e4 = dct[id]['energy']
        assert e1 == e2 == e3 == e4
        print(e1)
    ae = 2 * c.get(1).energy - c.get(2).energy
    assert abs(ae - 0.65376) < 1e-3
