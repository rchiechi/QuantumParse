from __future__ import print_function
"""This module defines an ASE interface to SIESTA.

Written by Mads Engelund
http://www.mads-engelund.net

Home of the SIESTA package:
http://www.uam.es/departamentos/ciencias/fismateriac/siesta
"""
from ase.calculators.siesta.base_siesta import BaseSiesta


# Version 3.2 of Siesta
class Siesta3_2(BaseSiesta):
    allowed_xc = {
        'LDA': ['PZ', 'CA', 'PW92'],
        'GGA': ['PBE', 'revPBE', 'RPBE',
                'WC', 'PBEsol', 'LYP']}

    unit_fdf_keywords = {
        'BasisPressure': 'eV/Ang**3',
        'LatticeConstant': 'Ang',
        'ZM.UnitsLength': 'Ang',
        'WarningMiniumAtomicDistance': 'Ang',
        'MaxBondDistance': 'Ang',
        'kgrid_cutoff': 'Ang',
        'DM.EnergyTolerance': 'eV',
        'DM.HarrisTolerance': 'eV',
        'EggboxScale': 'eV',
        'ElectronicTemperature': 'eV',
        'OMM.TPreconScale': 'eV',
        'ON.eta': 'eV',
        'ON.eta_alpha': 'eV',
        'ON.eta_beta': 'eV',
        'ON.RcLWF': 'Ang',
        'ON.ChemicalPotentialRc': 'Ang',
        'ON.ChemicalPotentialTemperature': 'eV',
        'Optical.EnergyMinimum': 'eV',
        'Optical.EnergyMaximum': 'eV',
        'Optical.Broaden': 'eV',
        'Optical.Scissor': 'eV',
        'MD.MaxForceTol': 'eV/Ang',
        'MD.MaxStressTol': 'eV/Ang**3',
        'MD.MaxCGDispl': 'Ang',
        'MD.PreconditioningVariableCell': 'Ang',
        'ZM.ForceTolLength': 'eV/Ang',
        'ZM.ForceTolAngle': 'eV/rad',
        'ZM.MaxDiplLength': 'Ang',
        'MD.FIRE.TimeStep': 's',
        'MD.TargetPressure': 'eV/Ang**3',
        'MD.LengthTimeStep': 's',
        'MD.InitialTemperature': 'eV',
        'MD.TargetTemperature': 'eV',
        'MD.NoseMass': 'Kg*m**2',               # Not in ASE unit
        'MD.ParrinelloRahmanMass': 'Kg*m**2',   # Not in ASE unit
        'MD.TauRelax': 's',
        'MD.BulkModulus': 'eV/Ang**3',
        'MD.FCDispl': 'Ang'}

    allowed_fdf_keywords = [
        'MD.UseSaveXV',
        'MD.UseSaveCG',
        'LongOutput',
        'SystemName',
        'SystemLabel',
        'NumberOfSpecies',
        'NumberOfAtoms',
        'ChemicalSpeciesLabel',
        'AtomicMass',
        'PAO.BasisType',
        'PAO.SplitNorm',
        'PAO.SplitNormH',
        'PAO.NewSplitCode',
        'PAO.FixSplitTable',
        'PAO.FixSplitTailNorm',
        'PAO.SoftDefault',
        'PAO.SoftInnerRadius',
        'PAO.SoftPotential',
        'PS.lmax',
        'PS.KBprojectors',
        'FilterCutoff',
        'FilterTol',
        'User.Basis',
        'User.Basis.NetCDF',
        'BasisPressure',
        'ReparametrizePseudos',
        'New.A.Parameter',
        'New.B.Parameter',
        'Rmax.Radial.Grid',
        'Restricted.Radial.Grid',
        'LatticeConstant',
        'LatticeParameters',
        'LatticeVectors',
        'SuperCell',
        'AtomicCoordinatesFormat',
        'AtomicCoorFormatOut',
        'AtomicCoordinatesOrigin',
        'AtomicCoordinatesAndAtomicSpecies',
        'Zmatrix',
        'ZM.UnitsLength',
        'ZM.UnitsAngle',
        'WriteCoorXmol',
        'WriteCoorCerius',
        'WriteMDXmol',
        'WarningMiniumAtomicDistance',
        'MaxBondDistance',
        'kgrid_cutoff',
        'kgrid_Monkhorst_Pack',
        'ChangeKgridInMD',
        'TimeReversalSymmetryForKpoints',
        'WriteKpoints',
        'XC.functional',
        'XC.authors',
        'XC.hydrid',
        'SpinPolarised',
        'NonCollinearSpin',
        'FixSpin',
        'TotalSpin',
        'SingleExcitation',
        'Harris_functional',
        'MaxSCFIterations',
        'SCFMustConverge',
        'DM.MixingWeight',
        'DM.NumberPulay',
        'DM.Pulay.Avoid.First.After.Kick',
        'DM.NumberBroyden',
        'DM.Broyden.Cycle.On.Maxit',
        'DM.NumberKick',
        'DM.KickMixingWeight',
        'DM.MixSCF1',
        'DM.UseSaveDM',
        'DM.FormattedFiles',
        'DM.FormattedInput',
        'DM.FormattedOutput',
        'DM.InitSpinAF',
        'DM.InitSpin',
        'DM.AllowReuse',
        'DM.AllowExtrapolation',
        'SCF.Read.Charge.NetCDF',
        'SCF.ReadDeformation.Charge.NetCDF',
        'WriteDM',
        'WriteDM.NetCDF',
        'WrireDMHS.NetCDF',
        'WriteDM.History.NetCDF',
        'WrireDMHS.History.NetCDF',
        'DM.Tolerance',
        'DM.Require.Energy.Convergence',
        'DM.EnergyTolerance',
        'DM.Require.Harris.Convergence',
        'DM.Harris.Tolerance',
        'MeshCutoff',
        'MeshSubDivisions'
        'GridCellSampling',
        'EggboxRemove',
        'EggboxScale',
        'NeglNonOverlapInt',
        'SaveHS',
        'FixAuxiliaryCell',
        'NaiveAuxiliaryCell',
        'SolutionMethod',
        'NumberOfEigenStates',
        'Use.New.Diagk',
        'Diag.DivideAndConquer',
        'Diag.AllInOne',
        'Diag.NoExpert',
        'Diag.PreRotate',
        'Diag.Use2D',
        'WriteEigenvalues',
        'OccupationFunction',
        'OccupationMPOrder',
        'ElectronicTemperature',
        'OMM.UseCholesky',
        'OMM.Use2D',
        'OMM.UseSparse',
        'OMM.Precon',
        'OMM.PreconFirstStep',
        'OMM.Diagon',
        'OMM.DiagonFirstStep',
        'OMM.BlockSize',
        'OMM.TPreconScale',
        'OMM.RelTol',
        'OMM.Eigenvalues',
        'OMM.WriteCoeffs',
        'OMM.ReadCoeffs',
        'OMM.LongOutput',
        'ON.functional',
        'ON.MaxNumIter',
        'ON.etol',
        'ON.eta',
        'ON.eta_alpha',
        'ON.eta_beta',
        'ON.RcLWF',
        'ON.ChemicalPotential',
        'ON.ChemicalPotentialUse',
        'ON.ChemicalPotentialRc',
        'ON.ChemicalPotentialTemperature',
        'ON.ChemicalPotentialOrder',
        'ON.LowerMemory',
        'ON.UseSaveLWF',
        'BandLinesScale',
        'BandLines',
        'BandPoints',
        'WriteKbands',
        'WriteBands',
        'WFS.Write.For.Bands',
        'WFS.Band.Min',
        'WFS.Band.Max',
        'WaveFuncPointScale',
        'WaveFuncKPoints',
        'WriteWaveFunctions',
        'ProjectedDensityOfStates',
        'LocalDensityOfStates',
        'WriteMullikenPop',
        'MullikenInSCF',
        'WriteHirshfeldPop',
        'WriteVoronoiPop',
        'PartialChargesAtEveryGeometry',
        'PartialChargesAtEveryScfStep',
        'COOP.Write',
        'WFS.Energy.Min',
        'WFS.Energy.Max',
        'WFS.Band.Min',
        'WFS.Band.Max',
        'OpticalCalculation',
        'Optical.EnergyMinimum',
        'Optical.EnergyMaximum',
        'Optical.Broaden',
        'Optical.Scissor',
        'Optical.NumberOfBands',
        'Optical.Mesh',
        'Optical.OffsetMesh',
        'Optical.PolarizationType',
        'Optical.Vector',
        'PolarizationGrids',
        'BornCharge',
        'NetCharge',
        'SimulateDoping',
        'ExternalElectricField',
        'SlabDipoleCorrection',
        'SaveRho',
        'SaveDeltaRho',
        'SaveElectrostaticPotential',
        'SaveNeutralAtomPotential',
        'SaveTotalPotential',
        'SaveIonicCharge',
        'SaveTotalCharge',
        'SaveBaderCharge',
        'SaveInitialChargeDensity',
        'MM.Potentials',
        'MM.Cutoff',
        'MM.UnitsEnergy',
        'MM.UnitsDistance',
        'MM.Grimme.D',
        'MM.Grimme.S6',
        'BlockSize'
        'ProcessorY',
        'Diag.Memory',
        'Diag.ParallelOverK',
        'UseDomainDecomposition',
        'UseSpatialDecomposition',
        'RcSpatial',
        'DirectPhi',
        'AllocReportLevel',
        'TimerReportThreshold',
        'UserTreeTimer',
        'UseSaveData',
        'WriteDenchar',
        'MD.TypeOfRun',
        'MD.VariableCell',
        'MD.ConstantVolume',
        'MD.RelaxCellOnly',
        'MD.MaxForceTol',
        'MD.MaxStressTol',
        'MD.NumCGsteps',
        'MD.MaxCGDispl',
        'MD.PreconditioningVariableCell',
        'ZM.ForceTolLength',
        'ZM.ForceTolAngle',
        'ZM.MaxDiplLength',
        'ZM.MaxDiplAngle',
        'MD.UseSaveCG',
        'MD.Broyden.History.Steps',
        'MD.Broyden.Cycle.On.Maxit',
        'MD.Broyden.Initial.Inverse.Jacobian',
        'MD.FIRE.TimeStep',
        'MD.Quench',
        'MD.FireQuench',
        'MD.TargetPressure',
        'MD.TargetStress',
        'MD.RemoveIntramolecularPressure',
        'MD.InitialTimeStep',
        'MD.FinalTimestep',
        'MD.LengthTimeStep',
        'MD.InitialTemperature',
        'MD.TargetTemperature',
        'MD.NoseMass',
        'MD.ParrinelloRahmanMass',
        'MD.AnnealOption',
        'MD.TauRelax',
        'MD.BulkModulus',
        'WriteCoorInitial',
        'WriteCoorStep',
        'WriteForces',
        'WriteMDhistory',
        'GeometryConstraints',
        'MD.FCDispl',
        'MD.FCfirst',
        'MD.FClast',
        'PhononLabels',
        'MD.ATforPhonon']


# Trunk version, snapshot 462
class SiestaTrunk462(BaseSiesta):
    allowed_xc = {
        'LDA': ['PZ', 'CA', 'PW92'],
        'GGA': ['PW91', 'PBE', 'revPBE', 'RPBE',
                'WC', 'AM05', 'PBEsol', 'PBEJsJrLO',
                'PBEGcGxLO', 'PBEGcGxHEG', 'BLYP'],
        'VDW': ['DRSLL', 'LMKLL', 'KBM', 'C09', 'BH', 'VV']}
    unit_fdf_keywords = {
        'BasisPressure': 'eV/Ang**3',
        'LatticeConstant': 'Ang',
        'ZM.UnitsLength': 'Ang',
        'WarningMiniumAtomicDistance': 'Ang',
        'MaxBondDistance': 'Ang',
        'kgrid_cutoff': 'Ang',
        'DM.EnergyTolerance': 'eV',
        'DM.HarrisTolerance': 'eV',
        'EggboxScale': 'eV',
        'ElectronicTemperature': 'eV',
        'ON.eta': 'eV',
        'ON.eta_alpha': 'eV',
        'ON.eta_beta': 'eV',
        'ON.RcLWF': 'Ang',
        'ON.ChemicalPotentialRc': 'Ang',
        'ON.ChemicalPotentialTemperature': 'eV',
        'Optical.EnergyMinimum': 'eV',
        'Optical.EnergyMaximum': 'eV',
        'Optical.Broaden': 'eV',
        'Opticalcissor': 'eV',
        'MD.MaxForceTol': 'eV/Ang',
        'MD.MaxStressTol': 'eV/Ang**3',
        'MD.MaxCGDispl': 'Ang',
        'MD.PreconditioningVariableCell': 'Ang',
        'ZM.ForceTolLength': 'eV/Ang',
        'ZM.ForceTolAngle': 'eV/rad',
        'ZM.MaxDiplLength': 'Ang',
        'MD.FIRE.TimeStep': 's',
        'MD.TargetPressure': 'eV/Ang**3',
        'MD.LengthTimeStep': 's',
        'MD.InitialTemperature': 'eV',
        'MD.TargetTemperature': 'eV',
        'MD.NoseMass': 'Kg*m**2',               # Not in ASE unit
        'MD.ParrinelloRahmanMass': 'Kg*m**2',   # Not in ASE unit
        'MD.TauRelax': 's',
        'MD.BulkModulus': 'eV/Ang**3',
        'MD.FCDispl': 'Ang'}

    allowed_fdf_keywords = [
        'MD.UseSaveXV',
        'MD.UseSaveCG',
        'LongOutput',
        'SystemName',
        'SystemLabel',
        'NumberOfSpecies',
        'NumberOfAtoms',
        'ChemicalSpeciesLabel',
        'AtomicMass',
        'PAO.BasisType',
        'PAO.SplitNorm',
        'PAO.SplitNormH',
        'PAO.NewSplitCode',
        'PAO.FixSplitTable',
        'PAO.FixSplitTailNorm',
        'PAO.EnergyCutoff',
        'PAO.EnergyPolCutoff',
        'PAO.EnergyContractionCutoff',
        'PAO.SoftDefault',
        'PAO.SoftInnerRadius',
        'PAO.SoftPotential',
        'PS.lmax',
        'PS.KBprojectors',
        'KB.New.Reference.Orbitals',
        'PAO.Basis',
        'FilterCutoff',
        'FilterTol',
        'User.Basis',
        'User.Basis.NetCDF',
        'BasisPressure',
        'ReparametrizePseudos',
        'New.A.Parameter',
        'New.B.Parameter',
        'Rmax.Radial.Grid',
        'Restricted.Radial.Grid',
        'LatticeConstant',
        'LatticeParameters',
        'LatticeVectors',
        'SuperCell',
        'AtomicCoordinatesFormat',
        'AtomicCoorFormatOut',
        'AtomicCoordinatesOrigin',
        'AtomicCoordinatesAndAtomicSpecies',
        'Zmatrix',
        'ZM.UnitsLength',
        'ZM.UnitsAngle',
        'WriteCoorXmol',
        'WriteCoorCerius',
        'WriteMDXmol',
        'WarningMiniumAtomicDistance',
        'MaxBondDistance',
        'kgrid_cutoff',
        'kgrid_Monkhorst_Pack',
        'ChangeKgridInMD',
        'TimeReversalSymmetryForKpoints',
        'WriteKpoints',
        'XC.functional',
        'XC.authors',
        'XC.hydrid',
        'SpinPolarised',
        'NonCollinearSpin',
        'FixSpin',
        'TotalSpin',
        'SingleExcitation',
        'Harris_functional',
        'MinSCFIterations',
        'MaxSCFIterations',
        'SCFMustConverge',
        'MixHamiltonian',
        'DM.MixSCF1',
        'SCF.MixAfterConvergence',
        'DM.MixingWeight',
        'DM.NumberPulay',
        'SCF.PulayDamping',
        'SCF.PulayMinimumHistory',
        'SCF.PulayDmaxRegion',
        'DM.NumberKick',
        'DM.KickMixingWeight',
        'DM.Pulay.Avoid.First.After.Kick',
        'SCF.LinearMixingAfterPulay',
        'SCF.MixingWeightAfterPulay',
        'SCF.Pulay.UseSVD',
        'SCF.Pulay.DebugSVD',
        'SCF.Pulay.RcondSVD',
        'DM.PulayOnFile',
        'DM.NumberBroyden',
        'DM.Broyden.Cycle.On.Maxit',
        'DM.Broyden.Variable.Weight',
        'MixCharge',
        'SCF.Kerker.q0sq',
        'SCF.RhoGMixingCutoff',
        'SCF.RhoG.DIIS.Depth',
        'SCF.RhoG.Metric.Preconditioner.Cutoff',
        'SCF.DebugRhogMixing',
        'DebugDIIS',
        'SCF.MixCharge.SCF1',
        'DM.UseSaveDM',
        'DM.FormattedFiles',
        'DM.FormattedInput',
        'DM.FormattedOutput',
        'DM.InitSpinAF',
        'DM.InitSpin',
        'DM.AllowReuse',
        'DM.AllowExtrapolation',
        'SCF.Read.Charge.NetCDF',
        'SCF.ReadDeformation.Charge.NetCDF',
        'WriteDM',
        'WriteDM.NetCDF',
        'WrireDMHS.NetCDF',
        'WriteDM.History.NetCDF',
        'WrireDMHS.History.NetCDF',
        'DM.Tolerance',
        'DM.Require.Energy.Convergence',
        'DM.EnergyTolerance',
        'DM.Require.Harris.Convergence',
        'DM.Harris.Tolerance',
        'MeshCutoff',
        'MeshSubDivisions'
        'GridCellSampling',
        'EggboxRemove',
        'EggboxScale',
        'NeglNonOverlapInt',
        'SaveHS',
        'FixAuxiliaryCell',
        'NaiveAuxiliaryCell',
        'SolutionMethod',
        'NumberOfEigenStates',
        'Use.New.Diagk',
        'Diag.DivideAndConquer',
        'Diag.AllInOne',
        'Diag.NoExpert',
        'Diag.PreRotate',
        'Diag.Use2D',
        'WriteEigenvalues',
        'OccupationFunction',
        'OccupationMPOrder',
        'ElectronicTemperature',
        'ON.functional',
        'ON.MaxNumIter',
        'ON.etol',
        'ON.eta',
        'ON.eta_alpha',
        'ON.eta_beta',
        'ON.RcLWF',
        'ON.ChemicalPotential',
        'ON.ChemicalPotentialUse',
        'ON.ChemicalPotentialRc',
        'ON.ChemicalPotentialTemperature',
        'ON.ChemicalPotentialOrder',
        'ON.LowerMemory',
        'ON.UseSaveLWF',
        'BandLinesScale',
        'BandLines',
        'BandPoints',
        'WriteKbands',
        'WriteBands',
        'WFS.Write.For.Bands',
        'WFS.Band.Min',
        'WFS.Band.Max',
        'WaveFuncPointScale',
        'WaveFuncKPoints',
        'WriteWaveFunctions',
        'ProjectedDensityOfStates',
        'LocalDensityOfStates',
        'WriteMullikenPop',
        'MullikenInSCF',
        'WriteHirshfeldPop',
        'WriteVoronoiPop',
        'PartialChargesAtEveryGeometry',
        'PartialChargesAtEveryScfStep',
        'COOP.Write',
        'WFS.Energy.Min',
        'WFS.Energy.Max',
        'OpticalCalculation',
        'Optical.EnergyMinimum',
        'Optical.EnergyMaximum',
        'Optical.Broaden',
        'Optical.Scissor',
        'Optical.NumberOfBands',
        'Optical.Mesh',
        'Optical.OffsetMesh',
        'Optical.PolarizationType',
        'Optical.Vector',
        'PolarizationGrids',
        'BornCharge',
        'NetCharge',
        'SimulateDoping',
        'ExternalElectricField',
        'SlabDipoleCorrection',
        'SaveRho',
        'SaveDeltaRho',
        'SaveElectrostaticPotential',
        'SaveNeutralAtomPotential',
        'SaveBaderCharge',
        'SaveInitialChargeDensity',
        'MM.Potentials',
        'MM.Cutoff',
        'MM.UnitsEnergy',
        'MM.UnitsDistance',
        'MM.Grimme.D',
        'MM.Grimme.S6',
        'BlockSize'
        'ProcessorY',
        'Diag.Memory',
        'Diag.ParallelOverK',
        'UseDomainDecomposition',
        'UseSpatialDecomposition',
        'RcSpatial',
        'DirectPhi',
        'AllocReportLevel',
        'UseSaveData',
        'WriteDenchar',
        'MD.TypeOfRun',
        'MD.VariableCell',
        'MD.ConstantVolume',
        'MD.RelaxCellOnly',
        'MD.MaxForceTol',
        'MD.MaxStressTol',
        'MD.NumCGsteps',
        'MD.MaxCGDispl',
        'MD.PreconditioningVariableCell',
        'ZM.ForceTolLength',
        'ZM.ForceTolAngle',
        'ZM.MaxDiplLength',
        'ZM.MaxDiplAngle',
        'MD.UseSaveCG',
        'MD.Broyden.History.Steps',
        'MD.Broyden.Cycle.On.Maxit',
        'MD.Broyden.Initial.Inverse.Jacobian',
        'MD.FIRE.TimeStep',
        'MD.Quench',
        'MD.FireQuench',
        'MD.TargetPressure',
        'MD.TargetStress',
        'MD.RemoveIntramolecularPressure',
        'MD.InitialTimeStep',
        'MD.FinalTimestep',
        'MD.LengthTimeStep',
        'MD.InitialTemperature',
        'MD.TargetTemperature',
        'MD.NoseMass',
        'MD.ParrinelloRahmanMass',
        'MD.AnnealOption',
        'MD.TauRelax',
        'MD.BulkModulus',
        'WriteCoorInitial',
        'WriteCoorStep',
        'WriteForces',
        'WriteMDhistory',
        'GeometryConstraints',
        'MD.FCDispl',
        'MD.FCfirst',
        'MD.FClast',
        'PhononLabels',
        'MD.ATforPhonon']


# Define the default siesta version.
Siesta = Siesta3_2
