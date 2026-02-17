import torch
import torch.nn.functional as F
from torchmetrics import Metric
import numpy as np
from typing import Optional


class StructureAmplitudeLocation(Metric):
    """
    Structure-Amplitude-Location (SAL) metric for evaluating precipitation forecasts.

    SAL decomposes forecast errors into three components:
    - S (Structure): Measures how scattered/intense the precipitation objects are
    - A (Amplitude): Measures overall amplitude bias
    - L (Location): Measures location displacement error

    Args:
        threshold: Threshold value for defining precipitation events
        min_area: Minimum area for considering a precipitation object
    """

    def __init__(self, threshold: float = 0.0, min_area: int = 1):
        super().__init__()

        self.threshold = threshold
        self.min_area = min_area

        self.add_state("s_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("a_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("l_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

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
            pred = pred.view(-1, height, width)
            target = target.view(-1, height, width)
        else:
            batch_size, height, width = pred.shape

        batch_sal_values = []

        for i in range(pred.shape[0]):
            pred_single = pred[i].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            target_single = target[i].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

            sal_vals = self._compute_sal_single(pred_single, target_single)
            if sal_vals is not None:
                batch_sal_values.append(sal_vals)

        if batch_sal_values:
            batch_sal_tensor = torch.stack(batch_sal_values)
            self.s_sum += torch.sum(batch_sal_tensor[:, 0])
            self.a_sum += torch.sum(batch_sal_tensor[:, 1])
            self.l_sum += torch.sum(batch_sal_tensor[:, 2])
            self.total_batches += len(batch_sal_values)

    def _compute_sal_single(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute SAL for a single prediction-target pair.

        Args:
            pred: Single prediction tensor of shape (1, 1, H, W)
            target: Single target tensor of shape (1, 1, H, W)

        Returns:
            Tensor containing [S, A, L] values, or None if computation fails
        """
        # Ensure inputs are detached and moved to CPU for numpy operations
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        # Compute amplitude component (A)
        pred_total = np.sum(pred_np)
        target_total = np.sum(target_np)

        if target_total == 0:
            # If target has no precipitation, A is undefined
            # We'll return zeros for all components in this case
            return torch.tensor([0.0, 0.0, 0.0])

        # Amplitude bias
        a = (pred_total - target_total) / target_total
        a = np.clip(a, -1.0, 1.0)  # Clip to [-1, 1]

        # Compute structure components (S)
        s = self._compute_structure_component(pred_np, target_np)

        # Compute location component (L)
        l = self._compute_location_component(pred_np, target_np)

        return torch.tensor([abs(s), abs(a), abs(l)])

    def _compute_structure_component(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the structure component of SAL.

        This measures how scattered/intense the precipitation objects are.
        """
        # Find connected components in both fields above threshold
        labeled_pred, pred_n = self._label_components(pred > self.threshold)
        labeled_target, target_n = self._label_components(target > self.threshold)

        if pred_n == 0 and target_n == 0:
            return 0.0
        elif pred_n == 0 or target_n == 0:
            # One field has no objects, the other does
            return 1.0

        # Calculate normalized entropy of object sizes
        pred_areas = self._get_component_sizes(labeled_pred, pred_n)
        target_areas = self._get_component_sizes(labeled_target, target_n)

        # Filter out areas below minimum size
        pred_areas = pred_areas[pred_areas >= self.min_area]
        target_areas = target_areas[target_areas >= self.min_area]

        if len(pred_areas) == 0 and len(target_areas) == 0:
            return 0.0
        elif len(pred_areas) == 0 or len(target_areas) == 0:
            return 1.0

        # Calculate entropy-based measure of structure difference
        # Higher entropy means more uniform distribution of object sizes
        pred_entropy = self._entropy(pred_areas)
        target_entropy = self._entropy(target_areas)

        # Normalize entropies and calculate difference
        max_possible_entropy_pred = (
            np.log(len(pred_areas)) if len(pred_areas) > 1 else 0
        )
        max_possible_entropy_target = (
            np.log(len(target_areas)) if len(target_areas) > 1 else 0
        )

        norm_pred_entropy = (
            pred_entropy / max_possible_entropy_pred
            if max_possible_entropy_pred > 0
            else 0
        )
        norm_target_entropy = (
            target_entropy / max_possible_entropy_target
            if max_possible_entropy_target > 0
            else 0
        )

        s = norm_pred_entropy - norm_target_entropy
        return np.clip(s, -1.0, 1.0)

    def _compute_location_component(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the location component of SAL.

        This measures displacement error between forecast and observation.
        """
        # Calculate centroids of precipitation objects
        target_bin = target > self.threshold
        pred_bin = pred > self.threshold

        if not np.any(target_bin) or not np.any(pred_bin):
            return 0.0

        # Calculate centroids
        target_com = self._center_of_mass(target_bin.astype(int))
        pred_com = self._center_of_mass(pred_bin.astype(int))

        # Calculate distance between centroids
        height, width = target.shape
        max_dist = np.sqrt(height**2 + width**2)  # Maximum possible distance

        if max_dist == 0:
            return 0.0

        dist = np.sqrt(
            (target_com[0] - pred_com[0]) ** 2 + (target_com[1] - pred_com[1]) ** 2
        )

        # Normalize by maximum possible distance
        l = dist / max_dist
        return np.clip(l, 0.0, 1.0)

    def _label_components(self, binary_array: np.ndarray):
        """
        Label connected components in a binary array.
        """
        # Implementation of connected component labeling
        # Using a simple flood-fill approach
        labeled = np.zeros_like(binary_array, dtype=int)
        current_label = 0

        rows, cols = binary_array.shape

        for i in range(rows):
            for j in range(cols):
                if binary_array[i, j] and labeled[i, j] == 0:
                    current_label += 1
                    self._flood_fill(binary_array, labeled, i, j, current_label)

        return labeled, current_label

    def _flood_fill(
        self,
        binary_array: np.ndarray,
        labeled: np.ndarray,
        row: int,
        col: int,
        label: int,
    ):
        """
        Flood fill algorithm to label connected components.
        """
        rows, cols = binary_array.shape
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()

            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if not binary_array[r, c] or labeled[r, c] != 0:
                continue

            labeled[r, c] = label

            # Add neighbors to stack
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))

    def _get_component_sizes(self, labeled_array: np.ndarray, num_labels: int):
        """
        Get sizes of each connected component.
        """
        if num_labels == 0:
            return np.array([])

        sizes = np.bincount(labeled_array.flatten())[1 : num_labels + 1]
        return sizes

    def _center_of_mass(self, binary_array: np.ndarray):
        """
        Calculate center of mass of a binary array.
        """
        rows, cols = np.where(binary_array)
        if len(rows) == 0:
            return (0, 0)

        center_row = np.mean(rows)
        center_col = np.mean(cols)
        return (center_row, center_col)

    def _entropy(self, counts):
        """
        Calculate entropy of a distribution.
        """
        if len(counts) == 0:
            return 0.0

        probs = counts / np.sum(counts)
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log(probs))

    def compute(self):
        """
        Compute the average SAL components.

        Returns:
            Dictionary containing mean S, A, L values
        """
        if self.total_batches == 0:
            return {
                "structure": torch.tensor(float("nan")),
                "amplitude": torch.tensor(float("nan")),
                "location": torch.tensor(float("nan")),
            }

        avg_s = self.s_sum / self.total_batches
        avg_a = self.a_sum / self.total_batches
        avg_l = self.l_sum / self.total_batches

        return {"structure": avg_s, "amplitude": avg_a, "location": avg_l}


class StructureAmplitudeLocationMean(Metric):
    """
    Mean Structure-Amplitude-Location metric across batches.
    """

    def __init__(self, threshold: float = 0.0, min_area: int = 1):
        super().__init__()

        self.sal_metric = StructureAmplitudeLocation(threshold, min_area)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.sal_metric.update(pred, target)

    def compute(self):
        return self.sal_metric.compute()
