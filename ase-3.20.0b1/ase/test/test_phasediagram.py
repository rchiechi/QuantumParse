"""Test phasediagram code."""
import pytest
from ase.phasediagram import PhaseDiagram


def test_phasediagram():
    """Test example from docs."""
    refs = [('Cu', 0.0),
            ('Au', 0.0),
            ('CuAu2', -0.2),
            ('CuAu', -0.5),
            ('Cu2Au', -0.7)]
    pd = PhaseDiagram(refs)
    energy, indices, coefs = pd.decompose('Cu3Au')
    assert energy == pytest.approx(-0.7)
    assert (indices == [4, 0]).all()
    assert coefs == pytest.approx(1.0)
