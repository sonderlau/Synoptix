from typing import Tuple
import torch
from torchmetrics import Metric


class EquitableThreatScore(Metric):
    """
    Equitable Threat Score (ETS) metric for evaluating event forecasts.

    The ETS accounts for random hits by comparing the threat score to that
    expected from random forecasts of the same frequency.

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
        Compute the ETS score.

        Returns:
            ETS value between 0 and 1 (1 is perfect)
        """
        total = self.tp + self.fp + self.fn + self.tn

        if total == 0:
            return torch.tensor(float("nan"))

        # Calculate observed threat score (same as CSI)
        obs_ts = (
            self.tp / (self.tp + self.fp + self.fn)
            if (self.tp + self.fp + self.fn) > 0
            else torch.tensor(0.0)
        )

        # Calculate expected threat score from random forecasts
        # Expected True Positives = (TP+FN)*(TP+FP)/total
        expected_tp = (self.tp + self.fn) * (self.tp + self.fp) / total
        exp_ts = (
            expected_tp / (self.tp + self.fp + self.fn)
            if (self.tp + self.fp + self.fn) > 0
            else torch.tensor(0.0)
        )

        # Calculate ETS
        if (obs_ts - exp_ts) == 0 and (1 - exp_ts) == 0:
            ets = torch.tensor(1.0)
        elif (1 - exp_ts) == 0:
            ets = torch.tensor(0.0)
        else:
            ets = (obs_ts - exp_ts) / (1 - exp_ts)

        return torch.clamp(ets, min=0.0, max=1.0)


class EquitableThreatScoreMean(Metric):
    """
    Mean Equitable Threat Score across multiple thresholds.
    """

    def __init__(self, thresholds: Tuple[float, ...]):
        super().__init__()

        self.thresholds = thresholds
        self.metrics = [EquitableThreatScore(thresh) for thresh in thresholds]

        # Add states for each threshold
        for i, thresh in enumerate(thresholds):
            self.add_state(f"ets_{i}", default=torch.tensor(0.0), dist_reduce_fx="sum")
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
            temp_metric = EquitableThreatScore(self.thresholds[i])
            temp_metric.update(pred, target)
            ets_val = temp_metric.compute()

            # Only accumulate if we got a valid result
            if not torch.isnan(ets_val):
                setattr(self, f"ets_{i}", getattr(self, f"ets_{i}") + ets_val)
                setattr(self, f"count_{i}", getattr(self, f"count_{i}") + 1)

    def compute(self):
        """
        Compute the mean ETS across all thresholds.

        Returns:
            Mean ETS value
        """
        ets_values = []
        for i in range(len(self.thresholds)):
            if getattr(self, f"count_{i}") > 0:
                ets_values.append(
                    getattr(self, f"ets_{i}") / getattr(self, f"count_{i}")
                )

        if len(ets_values) == 0:
            return torch.tensor(float("nan"))

        return torch.mean(torch.stack(ets_values))
