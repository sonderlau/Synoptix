from __future__ import annotations

from typing import Type

import torch
from torch.nn import AvgPool2d
from torchmetrics import Metric

from ..core import flatten_spatiotemporal, safe_tensor


class BinaryEventMetric(Metric):
    metric_name = "BinaryEventMetric"

    def __init__(self, threshold: float, pooling_size: int = 1):
        super().__init__()

        if pooling_size <= 0:
            raise ValueError("pooling_size must be greater than 0.")

        self.threshold = threshold
        self.pooling_size = pooling_size

        if self.pooling_size > 1:
            self.avg_pool = AvgPool2d(
                kernel_size=self.pooling_size,
                stride=1,
                padding=1,
            )

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def prepare(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = flatten_spatiotemporal(pred, target)

        if self.pooling_size > 1:
            pred = self.avg_pool(pred)
            target = self.avg_pool(target)

        pred_events = (pred >= self.threshold).float()
        target_events = (target >= self.threshold).float()
        return pred_events, target_events

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_events, target_events = self.prepare(pred, target)

        self.tp += torch.sum(pred_events * target_events)
        self.fp += torch.sum(pred_events * (1 - target_events))
        self.fn += torch.sum((1 - pred_events) * target_events)
        self.tn += torch.sum((1 - pred_events) * (1 - target_events))

    def zero(self) -> torch.Tensor:
        return safe_tensor(0.0, self.tp)

    def one(self) -> torch.Tensor:
        return safe_tensor(1.0, self.tp)

    def nan(self) -> torch.Tensor:
        return safe_tensor(float("nan"), self.tp)

    def __str__(self):
        return f"{self.metric_name}@{self.threshold:.3f}"


class MeanOverThresholdMetric(Metric):
    metric_cls: Type[BinaryEventMetric]
    metric_name = "MeanMetric"

    def __init__(self, thresholds: tuple[float, ...], **metric_kwargs):
        super().__init__()

        self.thresholds = thresholds
        self.metric_kwargs = metric_kwargs

        self.add_state(
            "score_sum",
            default=torch.zeros(len(thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "score_count",
            default=torch.zeros(len(thresholds)),
            dist_reduce_fx="sum",
        )

    def build_metric(self, threshold: float) -> BinaryEventMetric:
        return self.metric_cls(threshold=threshold, **self.metric_kwargs)

    def is_valid(self, score: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(score)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for index, threshold in enumerate(self.thresholds):
            metric = self.build_metric(threshold)
            metric.update(pred, target)
            score = metric.compute()

            if self.is_valid(score):
                self.score_sum[index] += score
                self.score_count[index] += 1

    def compute(self):
        valid = self.score_count > 0
        if not torch.any(valid):
            return torch.tensor(float("nan"))

        scores = self.score_sum[valid] / self.score_count[valid]
        return scores.mean()

    def __str__(self):
        return f"{self.metric_name}@Mean"
