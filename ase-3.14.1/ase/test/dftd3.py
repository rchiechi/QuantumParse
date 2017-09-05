import numpy as np
import os

from ase.calculators.dftd3 import DFTD3
from ase.test import NotAvailable
from ase.data.s22 import create_s22_system
from ase.build import bulk

releps = 1e-6
abseps = 1e-8

def close(val, reference, releps=releps, abseps=abseps):
    assert np.abs(val - reference) < max(np.abs(releps * reference), abseps)

def array_close(val, reference, releps=releps, abseps=abseps):
    valflat = val.flatten()
    refflat = reference.flatten()
    for i, vali in enumerate(valflat):
        close(vali, refflat[i], releps, abseps)

def main():
    if "ASE_DFTD3_COMMAND" not in os.environ:
        raise NotAvailable('$ASE_DFTD3_COMMAND not defined')
    
    # do all non-periodic calculations with Adenine-Thymine complex
    system = create_s22_system('Adenine-thymine_complex_stack')

    # Default is D3(zero)
    system.set_calculator(DFTD3())
    close(system.get_potential_energy(), -0.6681154466652238)

    # Only check forces once, for the default settings.
    f_ref = np.array([[ 0.0088385621657399, -0.0118387210205813, -0.0143242057174889],
                      [-0.0346912282737323,  0.0177797757792533, -0.0442349785529711],
                      [ 0.0022759961575945, -0.0087458217241648, -0.0051887171699909],
                      [-0.0049317224619103, -0.0215152368018880, -0.0062290998430756],
                      [-0.0013032612752381, -0.0356240144088481,  0.0203401124180720],
                      [-0.0110305568118348, -0.0182773178473497, -0.0023730575217145],
                      [ 0.0036258610447203, -0.0074994162928053, -0.0144058177906650],
                      [ 0.0005289754841564, -0.0035901842246731, -0.0103580836569947],
                      [ 0.0051775352510856, -0.0051076755874038, -0.0103428268442285],
                      [ 0.0011299493448658, -0.0185829345539878, -0.0087205807334006],
                      [ 0.0128459160503721, -0.0248356605575975,  0.0007946691695359],
                      [-0.0063194401470256, -0.0058117310787239, -0.0067932156139914],
                      [ 0.0013749100498893, -0.0118259631230572, -0.0235404547526578],
                      [ 0.0219558160992901, -0.0087512938555865, -0.0226017156485839],
                      [ 0.0001168268736984, -0.0138384169778581, -0.0014850073023105],
                      [ 0.0037893625607261,  0.0117649062330659,  0.0162375798918204],
                      [ 0.0011352730068862,  0.0142002748861793,  0.0129337874676760],
                      [-0.0049945288501837,  0.0073929058490670,  0.0088391871214417],
                      [ 0.0039715118075548,  0.0186949615105239,  0.0114822052853407],
                      [-0.0008003587963147,  0.0161735976004718,  0.0050357997715004],
                      [-0.0033142342134453,  0.0153658921418049, -0.0026233088963388],
                      [-0.0025451124688653,  0.0067994927521733, -0.0017127589489137],
                      [-0.0010451311609669,  0.0067173068779992,  0.0044413725566098],
                      [-0.0030829302438095,  0.0112138539867057,  0.0151213034444885],
                      [ 0.0117240581287903,  0.0161749855643631,  0.0173269837053235],
                      [-0.0025949288306356,  0.0158830629834040,  0.0155589787340858],
                      [ 0.0083784268665834,  0.0082132824775010,  0.0090603749323848],
                      [-0.0019694065480327,  0.0115576523485515,  0.0083901101633852],
                      [-0.0020036820791533,  0.0109276020920431,  0.0204922407855956],
                      [-0.0062424587308054,  0.0069848349714167,  0.0088791235460659]])
    
    array_close(system.get_forces(), f_ref)
    
    # calculate numerical forces, but use very loose comparison criteria!
    # dftd3 doesn't print enough digits to stdout to get good convergence
    f_numer = system.calc.calculate_numerical_forces(system, d=1e-4)
    array_close(f_numer, f_ref, releps=1e-2, abseps=1e-3)

    # D2
    system.set_calculator(DFTD3(old=True))
    close(system.get_potential_energy(), -0.8923443424663762)

    # D3(BJ)
    system.set_calculator(DFTD3(damping='bj'))
    close(system.get_potential_energy(), -1.211193213979179)

    # D3(zerom)
    system.set_calculator(DFTD3(damping='zerom'))
    close(system.get_potential_energy(), -2.4574447613705717)

    # D3(BJm)
    system.set_calculator(DFTD3(damping='bjm'))
    close(system.get_potential_energy(), -1.4662085277005799)

    # alternative tz parameters
    system.set_calculator(DFTD3(tz=True))
    close(system.get_potential_energy(), -0.6160295884482619)

    # D3(zero, ABC)
    system.set_calculator(DFTD3(abc=True))
    close(system.get_potential_energy(), -0.6528640090262864)

    # D3(zero) with revpbe parameters
    system.set_calculator(DFTD3(xc='revpbe'))
    close(system.get_potential_energy(), -1.5274869363442936)

    # Custom damping parameters
    system.set_calculator(DFTD3(s6=1.1, sr6=1.1, s8=0.6, sr8=0.9,
                                alpha6=13.0))
    close(system.get_potential_energy(), -1.082846357973487)

    # A couple of combinations, but not comprehensive

    # D3(BJ, ABC)
    system.set_calculator(DFTD3(damping='bj', abc=True))
    close(system.get_potential_energy(), -1.1959417763402416)

    # D3(zerom) with B3LYP parameters
    system.set_calculator(DFTD3(damping='zerom', xc='b3-lyp'))
    close(system.get_potential_energy(), -1.3369234231047677)

    # use diamond for bulk system
    system = bulk('C')
    
    system.set_calculator(DFTD3())
    close(system.get_potential_energy(), -0.2160072476277501)
    
    # Do one stress for the default settings
    s_ref = np.array([ 0.0182329043326,
                       0.0182329043326,
                       0.0182329043326,
                      -3.22757439831e-14,
                      -3.22766949320e-14,
                      -3.22766949320e-14])

    array_close(system.get_stress(), s_ref)
    
    # As with numerical forces, numerical stresses will not be very well
    # converged due to the limited number of digits printed to stdout
    # by dftd3. So, use very loose comparison criteria.
    s_numer = system.calc.calculate_numerical_stress(system, d=1e-4)
    array_close(s_numer, s_ref, releps=1e-2, abseps=1e-3)

main()
