import numpy as np
import torch
from torchmetrics import Metric

from ..core import flatten_spatiotemporal


class StructureAmplitudeLocation(Metric):
    def __init__(self, threshold: float = 0.0, min_area: int = 1):
        super().__init__()

        self.threshold = threshold
        self.min_area = min_area

        self.add_state("s_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("a_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("l_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = flatten_spatiotemporal(pred, target)

        batch_values = []
        for index in range(pred.shape[0]):
            value = self._compute_sal_single(pred[index], target[index])
            if value is not None:
                batch_values.append(value)

        if batch_values:
            batch_tensor = torch.stack(batch_values)
            self.s_sum += torch.sum(batch_tensor[:, 0])
            self.a_sum += torch.sum(batch_tensor[:, 1])
            self.l_sum += torch.sum(batch_tensor[:, 2])
            self.total_samples += len(batch_values)

    def _compute_sal_single(self, pred: torch.Tensor, target: torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        target_total = np.sum(target_np)
        if target_total == 0:
            return torch.tensor([0.0, 0.0, 0.0])

        pred_total = np.sum(pred_np)
        amplitude = np.clip((pred_total - target_total) / target_total, -1.0, 1.0)
        structure = self._compute_structure_component(pred_np, target_np)
        location = self._compute_location_component(pred_np, target_np)

        return torch.tensor([abs(structure), abs(amplitude), abs(location)])

    def _compute_structure_component(self, pred: np.ndarray, target: np.ndarray):
        labeled_pred, pred_count = self._label_components(pred > self.threshold)
        labeled_target, target_count = self._label_components(target > self.threshold)

        if pred_count == 0 and target_count == 0:
            return 0.0
        if pred_count == 0 or target_count == 0:
            return 1.0

        pred_areas = self._get_component_sizes(labeled_pred, pred_count)
        target_areas = self._get_component_sizes(labeled_target, target_count)

        pred_areas = pred_areas[pred_areas >= self.min_area]
        target_areas = target_areas[target_areas >= self.min_area]

        if len(pred_areas) == 0 and len(target_areas) == 0:
            return 0.0
        if len(pred_areas) == 0 or len(target_areas) == 0:
            return 1.0

        pred_entropy = self._entropy(pred_areas)
        target_entropy = self._entropy(target_areas)

        pred_norm = pred_entropy / np.log(len(pred_areas)) if len(pred_areas) > 1 else 0.0
        target_norm = (
            target_entropy / np.log(len(target_areas)) if len(target_areas) > 1 else 0.0
        )

        return np.clip(pred_norm - target_norm, -1.0, 1.0)

    def _compute_location_component(self, pred: np.ndarray, target: np.ndarray):
        pred_binary = pred > self.threshold
        target_binary = target > self.threshold

        if not np.any(pred_binary) or not np.any(target_binary):
            return 0.0

        pred_center = self._center_of_mass(pred_binary.astype(int))
        target_center = self._center_of_mass(target_binary.astype(int))

        height, width = target.shape
        max_distance = np.sqrt(height ** 2 + width ** 2)
        if max_distance == 0:
            return 0.0

        distance = np.sqrt(
            (target_center[0] - pred_center[0]) ** 2
            + (target_center[1] - pred_center[1]) ** 2
        )
        return np.clip(distance / max_distance, 0.0, 1.0)

    def _label_components(self, binary_array: np.ndarray):
        labeled = np.zeros_like(binary_array, dtype=int)
        current_label = 0

        rows, cols = binary_array.shape
        for row in range(rows):
            for col in range(cols):
                if binary_array[row, col] and labeled[row, col] == 0:
                    current_label += 1
                    self._flood_fill(binary_array, labeled, row, col, current_label)

        return labeled, current_label

    def _flood_fill(
        self,
        binary_array: np.ndarray,
        labeled: np.ndarray,
        row: int,
        col: int,
        label: int,
    ):
        rows, cols = binary_array.shape
        stack = [(row, col)]

        while stack:
            cur_row, cur_col = stack.pop()
            if cur_row < 0 or cur_row >= rows or cur_col < 0 or cur_col >= cols:
                continue
            if not binary_array[cur_row, cur_col] or labeled[cur_row, cur_col] != 0:
                continue

            labeled[cur_row, cur_col] = label
            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((cur_row + delta_row, cur_col + delta_col))

    def _get_component_sizes(self, labeled_array: np.ndarray, num_labels: int):
        if num_labels == 0:
            return np.array([])
        return np.bincount(labeled_array.flatten())[1 : num_labels + 1]

    def _center_of_mass(self, binary_array: np.ndarray):
        rows, cols = np.where(binary_array)
        if len(rows) == 0:
            return 0.0, 0.0
        return np.mean(rows), np.mean(cols)

    def _entropy(self, counts: np.ndarray):
        if len(counts) == 0:
            return 0.0
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def compute(self):
        if self.total_samples == 0:
            nan = torch.tensor(float("nan"))
            return {"structure": nan, "amplitude": nan, "location": nan}

        return {
            "structure": self.s_sum / self.total_samples,
            "amplitude": self.a_sum / self.total_samples,
            "location": self.l_sum / self.total_samples,
        }

    def __str__(self):
        return "SAL"


class StructureAmplitudeLocationMean(Metric):
    def __init__(self, threshold: float = 0.0, min_area: int = 1):
        super().__init__()
        self.metric = StructureAmplitudeLocation(threshold=threshold, min_area=min_area)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target)

    def compute(self):
        return self.metric.compute()

    def __str__(self):
        return "SALMean"
