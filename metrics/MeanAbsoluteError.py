import torch
from torchmetrics import Metric


class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mae", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")


    def update(self, pred: torch.Tensor, target: torch.Tensor):

        pred = pred.float()
        target = target.float()

        self.mae += torch.abs(target - pred).mean()
        self.count += 1


    def compute(self):
        return self.mae / self.count