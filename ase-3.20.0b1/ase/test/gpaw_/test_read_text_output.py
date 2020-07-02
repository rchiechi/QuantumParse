from ase.calculators.singlepoint import SinglePointDFTCalculator


def test_read_gpaw_out(datadir):
    """Test reading of gpaw text output"""
    from ase import io

    # read input

    output_file_name = datadir / 'gpaw_expected_text_output'
    atoms = io.read(output_file_name)

    # test calculator
    
    calc = atoms.calc
    assert isinstance(calc, SinglePointDFTCalculator)
    assert calc.name == 'vdwtkatchenko09prl'
    assert calc.parameters['calculator'] == 'gpaw'

    for contribution in [
            'kinetic', 'potential', 'external', 'xc',
            'entropy (-st)', 'local']:
        assert contribution in calc.energy_contributions


# for the record, include somehow XXX
# output in datadir / 'gpaw_expected_text_output' written by
"""
if 1:
    from ase.build import molecule
    from ase.calculators.vdwcorrection import vdWTkatchenko09prl

    from gpaw import GPAW, FermiDirac
    from gpaw.cluster import Cluster
    from gpaw.analyse.vdwradii import vdWradii
    from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning

    atoms = Cluster(molecule('H2'))
    atoms.minimal_box(3)
    
    calc = GPAW(xc='PBE', occupations=FermiDirac(0.1))
    atoms.calc = vdWTkatchenko09prl(
        HirshfeldPartitioning(calc),
        vdWradii(atoms.get_chemical_symbols(), 'PBE'))
    atoms.get_potential_energy()
"""
