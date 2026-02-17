from typing import Tuple
import torch
from torchmetrics import Metric


class ProbabilityOfDetection(Metric):
    """
    Probability of Detection (POD) metric for evaluating event forecasts.

    The POD measures the fraction of observed events that were correctly forecast.
    Also known as hit rate or recall/sensitivity.

    Args:
        threshold: Threshold value for defining events
    """

    def __init__(self, threshold: float):
        super().__init__()

        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

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
        self.fn += torch.sum((1 - pred_events) * target_events)

    def compute(self):
        """
        Compute the POD score.

        Returns:
            POD value between 0 and 1 (1 is perfect detection)
        """
        total_observed_events = self.tp + self.fn

        if total_observed_events == 0:
            return torch.tensor(1.0)  # Perfect if no events occurred

        pod = self.tp / total_observed_events

        return pod


class ProbabilityOfDetectionMean(Metric):
    """
    Mean Probability of Detection across multiple thresholds.
    """

    def __init__(self, thresholds: Tuple[float, ...]):
        super().__init__()

        self.thresholds = thresholds
        self.metrics = [ProbabilityOfDetection(thresh) for thresh in thresholds]

        # Add states for each threshold
        for i, thresh in enumerate(thresholds):
            self.add_state(
                f"pod_sum_{i}", default=torch.tensor(0.0), dist_reduce_fx="sum"
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
            temp_metric = ProbabilityOfDetection(self.thresholds[i])
            temp_metric.update(pred, target)
            pod_val = temp_metric.compute()

            # Only accumulate if we got a valid result
            if not torch.isnan(pod_val):
                setattr(self, f"pod_sum_{i}", getattr(self, f"pod_sum_{i}") + pod_val)
                setattr(self, f"count_{i}", getattr(self, f"count_{i}") + 1)

    def compute(self):
        """
        Compute the mean POD across all thresholds.

        Returns:
            Mean POD value
        """
        pod_values = []
        for i in range(len(self.thresholds)):
            if getattr(self, f"count_{i}") > 0:
                pod_values.append(
                    getattr(self, f"pod_sum_{i}") / getattr(self, f"count_{i}")
                )

        if len(pod_values) == 0:
            return torch.tensor(float("nan"))

        return torch.mean(torch.stack(pod_values))
