from .base import BinaryEventMetric, MeanOverThresholdMetric


class HeidkeSkillScore(BinaryEventMetric):
    metric_name = "HSS"

    def compute(self, eps: float = 1e-6):
        numerator = 2 * (self.tp * self.tn - self.fp * self.fn)
        denominator = (self.tp + self.fn) * (self.fn + self.tn)
        denominator += (self.tp + self.fp) * (self.fp + self.tn)
        return numerator.float() / (denominator.float() + eps)


class HeidkeSkillScoreMean(MeanOverThresholdMetric):
    metric_cls = HeidkeSkillScore
    metric_name = "HSS"

    def __init__(self, thresholds: tuple[float, ...]):
        super().__init__(thresholds=thresholds)
