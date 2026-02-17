from typing import Tuple
import torch
from torchmetrics import Metric


class FrequencyBias(Metric):
    """
    Frequency Bias (BIAS) metric for evaluating event forecasts.

    The BIAS measures the ratio of events forecast to events observed.
    A value of 1 indicates perfect reliability, >1 indicates overforecasting,
    and <1 indicates underforecasting.

    Args:
        threshold: Threshold value for defining events
    """

    def __init__(self, threshold: float):
        super().__init__()

        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with new predictions and targets.

        Args:
            pred: Predicted values of shape (batch, height, width) or (batch, time, height, width)
            target: Target values of same shape as pred
        """
        # Handle both 3D and 4D tensors
        if pred.dim() == 4:
            pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
            target = target.reshape(-1, target.shape[-2], target.shape[-1])

        # Convert to binary fields based on threshold
        pred_events = (pred >= self.threshold).float()
        target_events = (target >= self.threshold).float()

        # Calculate confusion matrix components
        self.tp += torch.sum(pred_events * target_events)
        self.fp += torch.sum(pred_events * (1 - target_events))
        self.fn += torch.sum((1 - pred_events) * target_events)
        self.tn += torch.sum((1 - pred_events) * (1 - target_events))

    def compute(self):
        """
        Compute the BIAS score.

        Returns:
            BIAS value (1 is perfect, >1 overforecasting, <1 underforecasting)
        """
        forecast_events = self.tp + self.fp
        observed_events = self.tp + self.fn

        if observed_events == 0:
            # If no events observed, bias is undefined if forecasted events, else perfect
            return (
                torch.tensor(float("inf")) if forecast_events > 0 else torch.tensor(1.0)
            )

        bias = forecast_events / observed_events

        return bias


class FrequencyBiasMean(Metric):
    """
    Mean Frequency Bias across multiple thresholds.
    """

    def __init__(self, thresholds: Tuple[float, ...]):
        super().__init__()

        self.thresholds = thresholds
        self.metrics = [FrequencyBias(thresh) for thresh in thresholds]

        # Add states for each threshold
        for i, thresh in enumerate(thresholds):
            self.add_state(
                f"bias_sum_{i}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(f"count_{i}", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with new predictions and targets.

        Args:
            pred: Predicted values of shape (batch, height, width) or (batch, time, height, width)
            target: Target values of same shape as pred
        """
        for i, metric in enumerate(self.metrics):
            # Create a copy of the metric to avoid state interference
            temp_metric = FrequencyBias(self.thresholds[i])
            temp_metric.update(pred, target)
            bias_val = temp_metric.compute()

            # Only accumulate if we got a finite result
            if torch.isfinite(bias_val):
                setattr(
                    self, f"bias_sum_{i}", getattr(self, f"bias_sum_{i}") + bias_val
                )
                setattr(self, f"count_{i}", getattr(self, f"count_{i}") + 1)

    def compute(self):
        """
        Compute the mean BIAS across all thresholds.

        Returns:
            Mean BIAS value
        """
        bias_values = []
        for i in range(len(self.thresholds)):
            if getattr(self, f"count_{i}") > 0:
                bias_values.append(
                    getattr(self, f"bias_sum_{i}") / getattr(self, f"count_{i}")
                )

        if len(bias_values) == 0:
            return torch.tensor(float("nan"))

        return torch.mean(torch.stack(bias_values))
