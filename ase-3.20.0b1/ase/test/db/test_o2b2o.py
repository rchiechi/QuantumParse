import pickle
from ase.db.core import object_to_bytes, bytes_to_object
from ase.cell import Cell
import numpy as np


def test_o2b2o():
    for o1 in [1.0,
               {'a': np.zeros((2, 2), np.float32),
                'b': np.zeros((0, 2), int)},
               ['a', 42, True, None, np.nan, np.inf, 1j],
               Cell(np.eye(3)),
               {'a': {'b': {'c': np.ones(3)}}}]:
        p1 = pickle.dumps(o1)
        b1 = object_to_bytes(o1)
        o2 = bytes_to_object(b1)
        p2 = pickle.dumps(o2)
        print(o2)
        print(b1)
        print()
        assert p1 == p2, (o1, p1, p2, vars(o1), vars(p1), vars(p2))
