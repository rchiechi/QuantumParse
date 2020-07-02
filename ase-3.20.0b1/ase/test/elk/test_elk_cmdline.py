def test_elk_cmdline(cli):
    # warning! parameters are not converged - only an illustration!
    cli.shell("""ase build -x fcc -a 4.04 Al | \
    ase run elk -p \
    "tasks=0,kpts=1.5,rgkmax=5.0,tforce=True,smearing=(fermi-dirac,0.05)" """,
        'elk')
