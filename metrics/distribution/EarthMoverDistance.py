import numpy as np
import torch
from torchmetrics import Metric

from ..core import flatten_spatiotemporal


class EarthMoverDistance(Metric):
    def __init__(self, normalize: bool = True):
        super().__init__()

        self.normalize = normalize

        self.add_state("emd_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = flatten_spatiotemporal(pred, target)

        pred = torch.clamp(pred, min=0.0)
        target = torch.clamp(target, min=0.0)

        pred_sum = torch.sum(pred, dim=(1, 2), keepdim=True)
        target_sum = torch.sum(target, dim=(1, 2), keepdim=True)

        pred_norm = torch.where(pred_sum > 0, pred / pred_sum, pred)
        target_norm = torch.where(target_sum > 0, target / target_sum, target)

        for index in range(pred_norm.shape[0]):
            value = self._compute_emd_2d(
                pred_norm[index].detach().cpu().numpy(),
                target_norm[index].detach().cpu().numpy(),
            )
            self.emd_sum += torch.tensor(value, dtype=torch.float, device=self.emd_sum.device)
            self.total_samples += 1

    def _compute_emd_2d(self, pred: np.ndarray, target: np.ndarray):
        height, width = pred.shape

        coords = np.stack(np.mgrid[0:height, 0:width], axis=-1).astype(float)
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        coords_flat = coords.reshape(-1, 2)

        pred_center = np.average(coords_flat, weights=pred_flat, axis=0)
        target_center = np.average(coords_flat, weights=target_flat, axis=0)

        center_distance = np.linalg.norm(pred_center - target_center)

        pred_spread = np.sqrt(
            np.average(np.sum((coords_flat - pred_center) ** 2, axis=1), weights=pred_flat)
        )
        target_spread = np.sqrt(
            np.average(np.sum((coords_flat - target_center) ** 2, axis=1), weights=target_flat)
        )

        value = center_distance + abs(pred_spread - target_spread)
        if self.normalize:
            diagonal = np.sqrt(height ** 2 + width ** 2)
            if diagonal > 0:
                value /= diagonal
        return value

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(float("nan"))
        return self.emd_sum / self.total_samples

    def __str__(self):
        return "EMD"


class Wasserstein1Distance(Metric):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.metric = EarthMoverDistance(normalize=normalize)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target)

    def compute(self):
        return self.metric.compute()

    def __str__(self):
        return "W1"
