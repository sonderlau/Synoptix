import torch
from torchmetrics import Metric


class RootMeanSquaredError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("rmse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        error = torch.sqrt(torch.mean((pred - target) ** 2))
        self.rmse += error
        self.count += 1

    def compute(self):
        return self.rmse / self.count

    def __str__(self):
        return "RMSE"
