import torch

from .base import BinaryEventMetric, MeanOverThresholdMetric


class FrequencyBias(BinaryEventMetric):
    metric_name = "BIAS"

    def compute(self):
        forecast_events = self.tp + self.fp
        observed_events = self.tp + self.fn

        if observed_events == 0:
            if forecast_events > 0:
                return torch.tensor(float("inf"), device=self.tp.device)
            return self.one()

        return forecast_events / observed_events


class FrequencyBiasMean(MeanOverThresholdMetric):
    metric_cls = FrequencyBias
    metric_name = "BIAS"

    def __init__(self, thresholds: tuple[float, ...]):
        super().__init__(thresholds=thresholds)
