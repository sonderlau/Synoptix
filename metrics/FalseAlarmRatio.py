from typing import Tuple
import torch
from torchmetrics import Metric


class FalseAlarmRatio(Metric):
    """
    False Alarm Ratio (FAR) metric for evaluating event forecasts.

    The FAR measures the fraction of forecast events that did not occur.
    It indicates the reliability of the forecast system.

    Args:
        threshold: Threshold value for defining events
    """

    def __init__(self, threshold: float):
        super().__init__()

        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")

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

    def compute(self):
        """
        Compute the FAR score.

        Returns:
            FAR value between 0 and 1 (0 is perfect reliability)
        """
        total_forecasted_events = self.tp + self.fp

        if total_forecasted_events == 0:
            return torch.tensor(0.0)  # Perfect if no events were forecast

        far = self.fp / total_forecasted_events

        return far


class FalseAlarmRatioMean(Metric):
    """
    Mean False Alarm Ratio across multiple thresholds.
    """

    def __init__(self, thresholds: Tuple[float, ...]):
        super().__init__()

        self.thresholds = thresholds
        self.metrics = [FalseAlarmRatio(thresh) for thresh in thresholds]

        # Add states for each threshold
        for i, thresh in enumerate(thresholds):
            self.add_state(
                f"far_sum_{i}", default=torch.tensor(0.0), dist_reduce_fx="sum"
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
            temp_metric = FalseAlarmRatio(self.thresholds[i])
            temp_metric.update(pred, target)
            far_val = temp_metric.compute()

            # Only accumulate if we got a valid result
            if not torch.isnan(far_val):
                setattr(self, f"far_sum_{i}", getattr(self, f"far_sum_{i}") + far_val)
                setattr(self, f"count_{i}", getattr(self, f"count_{i}") + 1)

    def compute(self):
        """
        Compute the mean FAR across all thresholds.

        Returns:
            Mean FAR value
        """
        far_values = []
        for i in range(len(self.thresholds)):
            if getattr(self, f"count_{i}") > 0:
                far_values.append(
                    getattr(self, f"far_sum_{i}") / getattr(self, f"count_{i}")
                )

        if len(far_values) == 0:
            return torch.tensor(float("nan"))

        return torch.mean(torch.stack(far_values))
