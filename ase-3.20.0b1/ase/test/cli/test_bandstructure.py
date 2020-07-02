from ase.lattice import RHL
from pathlib import Path


def test_ase_bandstructure(cli, plt):
    lat = RHL(3., 70.0)
    path = lat.bandpath()
    bs = path.free_electron_band_structure()

    bs_path = Path('bs.json')
    bs.write(bs_path)

    fig_path = Path('bs.png')

    cli.ase(f'band-structure {bs_path} --output {fig_path}')
    # If the CLI tool gave a text output, we could verify it.
    assert fig_path.is_file()

# Note: We don't have proper testing of --points, --range etc.  We
# test here on JSON input but the tool is in principle supposed to
# work on other formats, too (only gpw though as of now though).
