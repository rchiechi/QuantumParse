"""
Test Placzek and Albrecht resonant Raman implementations
"""
import pytest

from ase.vibrations.placzek import Profeta
from ase.vibrations.albrecht import Albrecht
from ase.calculators.h2morse import H2Morse, H2MorseExcitedStatesAndCalculator


def test_compare_placzek_albrecht_intensities():
    atoms = H2Morse()
    name = 'rrmorse'
    pr = Profeta(atoms, H2MorseExcitedStatesAndCalculator,
                 approximation='Placzek',
                 gsname=name, exname=name,
                 overlap=lambda x, y: x.overlap(y),
                 txt=None)
    pr.run()

    om = 1
    pri, ali = 0, 0

    """Albrecht A and P-P are approximately equal"""

    pr.approximation = 'p-p'
    pri = pr.absolute_intensity(omega=om)[-1]
    al = Albrecht(atoms, H2MorseExcitedStatesAndCalculator,
                  gsname=name, exname=name, overlap=True,
                  approximation='Albrecht A', txt=None)
    ali = al.absolute_intensity(omega=om)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)
   
    """Albrecht B+C and Profeta are approximately equal"""

    pr.approximation = 'Profeta'
    pri = pr.absolute_intensity(omega=om)[-1]
    al.approximation = 'Albrecht BC'
    ali = al.absolute_intensity(omega=om)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)
   
    """Albrecht and Placzek are approximately equal"""
    
    pr.approximation = 'Placzek'
    pri = pr.absolute_intensity(omega=om)[-1]
    al.approximation = 'Albrecht'
    ali = al.absolute_intensity(omega=om)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)
