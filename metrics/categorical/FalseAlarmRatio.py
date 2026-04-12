from .base import BinaryEventMetric, MeanOverThresholdMetric


class FalseAlarmRatio(BinaryEventMetric):
    metric_name = "FAR"

    def compute(self):
        denominator = self.tp + self.fp
        if denominator == 0:
            return self.zero()
        return self.fp / denominator


class FalseAlarmRatioMean(MeanOverThresholdMetric):
    metric_cls = FalseAlarmRatio
    metric_name = "FAR"

    def __init__(self, thresholds: tuple[float, ...]):
        super().__init__(thresholds=thresholds)
