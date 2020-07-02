def test_fleur_cmdline(cli):
    cli.shell('ase build -x fcc -a 4.04 Al | ase run fleur -p kpts=3.0,xc=PBE',
              'fleur')
