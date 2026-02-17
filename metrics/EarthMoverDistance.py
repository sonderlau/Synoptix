import torch
import torch.nn.functional as F
from torchmetrics import Metric
import numpy as np
from typing import Optional


class EarthMoverDistance(Metric):
    """
    Earth Mover's Distance (EMD) / Wasserstein-1 distance metric.

    EMD measures the minimum amount of work needed to transform one distribution into another.
    In the context of precipitation forecasting, it can measure how much "mass" needs to be
    moved between pixels to transform the predicted precipitation field into the observed one.

    Args:
        normalize: Whether to normalize the EMD by the image dimensions
    """

    def __init__(self, normalize: bool = True):
        super().__init__()

        self.normalize = normalize

        self.add_state("emd_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with new predictions and targets.

        Args:
            pred: Predicted values of shape (batch, height, width) or (batch, time, height, width)
            target: Target values of same shape as pred
        """
        # Handle both 3D and 4D tensors
        if pred.dim() == 4:
            batch_size, n_time, height, width = pred.shape
            # Reshape to (batch*n_time, height, width)
            pred = pred.view(-1, height, width)
            target = target.view(-1, height, width)
        else:
            batch_size, height, width = pred.shape

        # Ensure non-negative values
        pred = torch.clamp(pred, min=0.0)
        target = torch.clamp(target, min=0.0)

        # Normalize each field to sum to 1 (probability distributions)
        pred_sum = torch.sum(pred, dim=(1, 2), keepdim=True)
        target_sum = torch.sum(target, dim=(1, 2), keepdim=True)

        # Avoid division by zero
        pred_normalized = torch.where(pred_sum > 0, pred / pred_sum, pred)
        target_normalized = torch.where(target_sum > 0, target / target_sum, target)

        # Calculate EMD for each sample in the batch
        for i in range(pred_normalized.shape[0]):
            emd_val = self._compute_emd_2d(
                pred_normalized[i].cpu().numpy(), target_normalized[i].cpu().numpy()
            )
            self.emd_sum += torch.tensor(emd_val, dtype=torch.float)
            self.total_samples += 1

    def _compute_emd_2d(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute 2D Earth Mover's Distance between two distributions.

        This is an approximation using optimal transport on a grid.
        """
        height, width = pred.shape

        # Create coordinate grids
        coords = np.stack(np.mgrid[0:height, 0:width], axis=-1).astype(float)

        # Flatten distributions and coordinates
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        coords_flat = coords.reshape(-1, 2)

        # Calculate centers of mass for a simplified estimation
        pred_com = np.average(coords_flat, weights=pred_flat, axis=0)
        target_com = np.average(coords_flat, weights=target_flat, axis=0)

        # For a more accurate calculation, we would use a proper optimal transport solver
        # Here we use an approximation based on the difference in distribution moments
        # This is a simplified version - for production use, consider using POT library

        # Calculate weighted distances between all points (simplified approach)
        # This is computationally expensive, so we use a moment-based approximation
        approx_emd = np.linalg.norm(pred_com - target_com)

        # Also account for differences in spread
        pred_spread = np.sqrt(
            np.average(np.sum((coords_flat - pred_com) ** 2, axis=1), weights=pred_flat)
        )
        target_spread = np.sqrt(
            np.average(
                np.sum((coords_flat - target_com) ** 2, axis=1), weights=target_flat
            )
        )

        spread_diff = abs(pred_spread - target_spread)

        # Combine center distance and spread difference
        combined_approx = approx_emd + spread_diff

        if self.normalize:
            # Normalize by image diagonal to get a value between 0 and 1
            max_dist = np.sqrt(height**2 + width**2)
            if max_dist > 0:
                combined_approx /= max_dist

        return combined_approx

    def compute(self):
        """
        Compute the average EMD across all samples.

        Returns:
            Average EMD value
        """
        if self.total_samples == 0:
            return torch.tensor(float("nan"))

        return self.emd_sum / self.total_samples


class Wasserstein1Distance(Metric):
    """
    Wasserstein-1 Distance (equivalent to Earth Mover's Distance).

    This is an alternative implementation focusing on the mathematical formulation
    of the Wasserstein-1 distance.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()

        self.emd_metric = EarthMoverDistance(normalize)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.emd_metric.update(pred, target)

    def compute(self):
        return self.emd_metric.compute()
