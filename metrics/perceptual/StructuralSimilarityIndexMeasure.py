import torch
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMMetric


class StructuralSimilarityIndexMeasure(Metric):
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

        self.add_state("ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.metric = SSIMMetric(data_range=data_range)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.ssim += self.metric(pred, target)
        self.count += 1

    def compute(self):
        return self.ssim / self.count

    def __str__(self):
        return "SSIM"
