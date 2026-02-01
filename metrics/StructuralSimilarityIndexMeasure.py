from typing import Tuple

import torch
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMCal

class StructuralSimilarityIndexMeasure(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("ssim", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")

        self.ssim_value = SSIMCal(data_range=1.0).to(self.device)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.ssim += self.ssim_value(pred, target)
        self.count += 1

    def compute(self):
        return self.ssim / self.count