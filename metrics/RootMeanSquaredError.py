import torch
from torchmetrics import Metric


class RootMeanSquaredError(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("rmse", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.rmse += torch.sqrt(torch.mean((
            pred - target
        ) ** 2))

        self.count += 1

    def compute(self):
        return self.rmse / self.count