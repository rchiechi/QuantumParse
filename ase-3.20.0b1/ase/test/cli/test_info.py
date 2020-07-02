def test_info(cli):
    assert 'numpy' in cli.ase('info')


def test_info_formats(cli):
    assert 'traj' in cli.ase('info --formats')


def test_info_calculators(cli):
    assert 'gpaw' in cli.ase('info --calculators')
