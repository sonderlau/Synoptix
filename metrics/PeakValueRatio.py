import torch
from torchmetrics import Metric


class PeakValueRatio(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("pred_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("truth_sum", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.pred_sum += pred.max()
        self.truth_sum += target.max()

    def compute(self):
        return self.pred_sum / self.truth_sum