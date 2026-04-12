from .base import BinaryEventMetric, MeanOverThresholdMetric


class ProbabilityOfDetection(BinaryEventMetric):
    metric_name = "POD"

    def compute(self):
        denominator = self.tp + self.fn
        if denominator == 0:
            return self.one()
        return self.tp / denominator


class ProbabilityOfDetectionMean(MeanOverThresholdMetric):
    metric_cls = ProbabilityOfDetection
    metric_name = "POD"

    def __init__(self, thresholds: tuple[float, ...]):
        super().__init__(thresholds=thresholds)
