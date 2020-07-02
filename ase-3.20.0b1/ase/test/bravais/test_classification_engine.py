import pytest
import numpy as np
from ase.geometry.bravais_type_engine import generate_niggli_op_table


ref_info = {
    'FCC': 1,
    'BCC': 1,
    'CUB': 1,
    'TET': 2,
    'BCT': 5,
    'HEX': 2,
    'ORC': 1,
    'ORCC': 5,
    'ORCF': 2,
    'ORCI': 4,
    'RHL': 3,
    #'MCL': 15,
    #'MCLC': 27,
    #'TRI': 19,
}


# We disable the three lattices that have infinite reductions.
# Maybe we can test those, but not today.
assert len(ref_info) == 14 - 3

def ref_info_iter():
    for key, val in ref_info.items():
        yield key, val

@pytest.mark.parametrize('lattice_name,ref_nops', ref_info_iter())
def test_generate_niggli_table(lattice_name, ref_nops):
    length_grid = np.logspace(-1, 1, 60)
    angle_grid = np.linspace(30, 120, 90 + 59)
    table = generate_niggli_op_table(lattices=[lattice_name],
                                     angle_grid=angle_grid,
                                     length_grid=length_grid)
    for key in table:
        print('{}: {}'.format(key, len(table[key])))

    mappings = table[lattice_name]
    assert len(mappings) == ref_nops
