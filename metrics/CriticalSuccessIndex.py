from typing import Tuple

import torch
from torch.nn import AvgPool2d
from torchmetrics import Metric


class CriticalSuccessIndex(Metric):

    def __init__(self, threshold: float, pooling_size: int = 1):
        super().__init__()

        self.threshold = threshold
        self.pooling = pooling_size

        assert pooling_size > 0, "Pooling size must be an integer and greater than 0."

        if self.pooling > 1:
            self.avg_pool = AvgPool2d(kernel_size=pooling_size, stride=1, padding=1)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):

        if self.pooling > 1:
            pred = self.avg_pool(pred)
            target = self.avg_pool(target)


        pred = (pred >= self.threshold).float()
        target = (target >= self.threshold).float()

        self.tp += torch.sum((pred == 1) & (target == 1))
        self.fp += torch.sum((pred == 1) & (target == 0))
        self.fn += torch.sum((pred == 0) & (target == 1))

    def compute(self):

        denominator = self.tp + self.fp + self.fn

        if denominator > 0.0:
            return self.tp / denominator
        else:
            return torch.tensor(0.0)


class CriticalSuccessIndexMean(Metric):
    def __init__(self, thresholds: Tuple[float, ...], pooling_size: int = 1):
        super().__init__()

        self.thresholds = thresholds
        self.pooling = pooling_size

        assert pooling_size > 0, "Pooling size must be an integer and greater than 0."

        if self.pooling > 1:
            self.avg_pool = AvgPool2d(kernel_size=pooling_size, stride=1, padding=1)


        self.add_state("tp", default=torch.zeros(len(thresholds)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(thresholds)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(thresholds)), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if self.pooling > 1:
            pred = self.avg_pool(pred)
            target = self.avg_pool(target)

        for i, thresh in enumerate(self.thresholds):
            p = (pred >= thresh).float()
            t = (target >= thresh).float()

            self.tp[i] += torch.sum((p == 1) & (t == 1))
            self.fp[i] += torch.sum((p == 1) & (t == 0))
            self.fn[i] += torch.sum((p == 0) & (t == 1))

    def compute(self, eps: float = 1e-6):
        csi_per_threshold = self.tp / (self.tp + self.fn + self.fp + eps)

        return csi_per_threshold.mean()