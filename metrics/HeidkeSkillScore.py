from typing import Tuple

import torch
from torchmetrics import Metric


class HeidkeSkillScore(Metric):

    def __init__(self, threshold: float):
        super().__init__()

        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        p = (pred >= self.threshold).long()
        t = (target >= self.threshold).long()

        self.tp += torch.sum((p == 1) & (t == 1))
        self.fp += torch.sum((p == 1) & (t == 0))
        self.fn += torch.sum((p == 0) & (t == 1))
        self.tn += torch.sum((p == 0) & (t == 0))

    def compute(self, eps: float = 1e-6):
        numerator = 2 * (self.tp * self.tn - self.fp * self.fn)

        denominator = (self.tp + self.fn) * (self.fn + self.tn) + \
                      (self.tp + self.fp) * (self.fp + self.tn)

        hss = numerator.float() / (denominator.float() + eps)

        return hss.mean()


class HeidkeSkillScoreMean(Metric):

    def __init__(self, thresholds: Tuple[float, ...]):
        super().__init__()

        self.thresholds = thresholds

        self.add_state("tp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for i, thresh in enumerate(self.thresholds):
            p = (pred >= thresh).long()
            t = (target >= thresh).long()

            self.tp[i] += torch.sum((p == 1) & (t == 1))
            self.fp[i] += torch.sum((p == 1) & (t == 0))
            self.fn[i] += torch.sum((p == 0) & (t == 1))
            self.tn[i] += torch.sum((p == 0) & (t == 0))

    def compute(self, eps: float = 1e-6):
        numerator = 2 * (self.tp * self.tn - self.fp * self.fn)

        denominator = (self.tp + self.fn) * (self.fn + self.tn) + \
                      (self.tp + self.fp) * (self.fp + self.tn)

        hss = numerator.float() / (denominator.float() + eps)

        return hss.mean()