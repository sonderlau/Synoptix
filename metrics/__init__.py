from .categorical import (
    CriticalSuccessIndex,
    CriticalSuccessIndexMean,
    EquitableThreatScore,
    EquitableThreatScoreMean,
    FalseAlarmRatio,
    FalseAlarmRatioMean,
    FrequencyBias,
    FrequencyBiasMean,
    HeidkeSkillScore,
    HeidkeSkillScoreMean,
    ProbabilityOfDetection,
    ProbabilityOfDetectionMean,
)
from .distribution import EarthMoverDistance, Wasserstein1Distance
from .extremes import PeakValueRatio
from .frequency import PowerSpectraDensityError
from .neighborhood import FractionsSkillScore, FractionsSkillScoreMean
from .object_based import StructureAmplitudeLocation, StructureAmplitudeLocationMean
from .perceptual import PSNR, StructuralSimilarityIndexMeasure
from .regression import MeanAbsoluteError, RootMeanSquaredError

CSI = CriticalSuccessIndex
CSIMean = CriticalSuccessIndexMean
ETS = EquitableThreatScore
ETSMean = EquitableThreatScoreMean
FAR = FalseAlarmRatio
FARMean = FalseAlarmRatioMean
BIAS = FrequencyBias
BIASMean = FrequencyBiasMean
HSS = HeidkeSkillScore
HSSMean = HeidkeSkillScoreMean
POD = ProbabilityOfDetection
PODMean = ProbabilityOfDetectionMean
EMD = EarthMoverDistance
Wasserstein = Wasserstein1Distance
FSS = FractionsSkillScore
FSSMean = FractionsSkillScoreMean
SAL = StructureAmplitudeLocation
SALMean = StructureAmplitudeLocationMean
SSIM = StructuralSimilarityIndexMeasure
MAE = MeanAbsoluteError
RMSE = RootMeanSquaredError
PVR = PeakValueRatio
PSD = PowerSpectraDensityError
PowerSpectralDensityError = PowerSpectraDensityError

__all__ = [
    "BIAS",
    "BIASMean",
    "CSI",
    "CSIMean",
    "CriticalSuccessIndex",
    "CriticalSuccessIndexMean",
    "EMD",
    "ETS",
    "ETSMean",
    "EarthMoverDistance",
    "EquitableThreatScore",
    "EquitableThreatScoreMean",
    "FAR",
    "FARMean",
    "FSS",
    "FSSMean",
    "FalseAlarmRatio",
    "FalseAlarmRatioMean",
    "FractionsSkillScore",
    "FractionsSkillScoreMean",
    "FrequencyBias",
    "FrequencyBiasMean",
    "HSS",
    "HSSMean",
    "HeidkeSkillScore",
    "HeidkeSkillScoreMean",
    "MAE",
    "MeanAbsoluteError",
    "POD",
    "PODMean",
    "PSD",
    "PSNR",
    "PVR",
    "PeakValueRatio",
    "PowerSpectraDensityError",
    "PowerSpectralDensityError",
    "ProbabilityOfDetection",
    "ProbabilityOfDetectionMean",
    "RMSE",
    "RootMeanSquaredError",
    "SAL",
    "SALMean",
    "SSIM",
    "StructuralSimilarityIndexMeasure",
    "StructureAmplitudeLocation",
    "StructureAmplitudeLocationMean",
    "Wasserstein",
    "Wasserstein1Distance",
]
