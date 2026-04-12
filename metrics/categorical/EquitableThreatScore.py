import torch

from .base import BinaryEventMetric, MeanOverThresholdMetric


class EquitableThreatScore(BinaryEventMetric):
    metric_name = "ETS"

    def compute(self):
        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            return self.nan()

        expected_tp = (self.tp + self.fn) * (self.tp + self.fp) / total
        denominator = self.tp + self.fp + self.fn - expected_tp
        if denominator == 0:
            return self.zero()

        value = (self.tp - expected_tp) / denominator
        return torch.clamp(value, min=0.0, max=1.0)


class EquitableThreatScoreMean(MeanOverThresholdMetric):
    metric_cls = EquitableThreatScore
    metric_name = "ETS"

    def __init__(self, thresholds: tuple[float, ...]):
        super().__init__(thresholds=thresholds)
