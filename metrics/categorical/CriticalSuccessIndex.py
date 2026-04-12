import torch

from .base import BinaryEventMetric, MeanOverThresholdMetric


class CriticalSuccessIndex(BinaryEventMetric):
    metric_name = "CSI"

    def compute(self):
        denominator = self.tp + self.fp + self.fn
        if denominator > 0:
            return self.tp / denominator
        return self.zero()


class CriticalSuccessIndexMean(MeanOverThresholdMetric):
    metric_cls = CriticalSuccessIndex
    metric_name = "CSI"

    def __init__(self, thresholds: tuple[float, ...], pooling_size: int = 1):
        super().__init__(thresholds=thresholds, pooling_size=pooling_size)
