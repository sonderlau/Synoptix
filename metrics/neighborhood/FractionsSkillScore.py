from typing import Tuple

import torch
from torchmetrics import Metric

from ..core import flatten_spatiotemporal


class FractionsSkillScore(Metric):
    def __init__(
        self,
        threshold: float,
        window_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        reduce_zero_forecast: str = "skip",
    ):
        super().__init__()

        self.threshold = threshold
        self.window_sizes = window_sizes
        self.reduce_zero_forecast = reduce_zero_forecast

        for window_size in window_sizes:
            self.add_state(
                f"obs_sq_sum_{window_size}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"fcst_sq_sum_{window_size}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"obs_fcst_sum_{window_size}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"num_grids_{window_size}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = flatten_spatiotemporal(pred, target)

        obs_binary = (target >= self.threshold).float()
        fcst_binary = (pred >= self.threshold).float()

        for window_size in self.window_sizes:
            obs_fraction = self._compute_fractions(obs_binary, window_size)
            fcst_fraction = self._compute_fractions(fcst_binary, window_size)

            if self.reduce_zero_forecast == "skip":
                mask = fcst_fraction != 0
                obs_fraction = obs_fraction[mask]
                fcst_fraction = fcst_fraction[mask]

            setattr(
                self,
                f"obs_sq_sum_{window_size}",
                getattr(self, f"obs_sq_sum_{window_size}") + torch.sum(obs_fraction ** 2),
            )
            setattr(
                self,
                f"fcst_sq_sum_{window_size}",
                getattr(self, f"fcst_sq_sum_{window_size}") + torch.sum(fcst_fraction ** 2),
            )
            setattr(
                self,
                f"obs_fcst_sum_{window_size}",
                getattr(self, f"obs_fcst_sum_{window_size}")
                + torch.sum(obs_fraction * fcst_fraction),
            )
            setattr(
                self,
                f"num_grids_{window_size}",
                getattr(self, f"num_grids_{window_size}")
                + torch.tensor(obs_fraction.numel(), dtype=torch.float, device=pred.device),
            )

    def _compute_fractions(self, binary_field: torch.Tensor, window_size: int):
        if window_size == 1:
            return binary_field

        unfolded = binary_field.unfold(1, window_size, 1).unfold(2, window_size, 1)
        return unfolded.mean(dim=(-2, -1))

    def compute(self):
        results = {}
        for window_size in self.window_sizes:
            obs_sq_sum = getattr(self, f"obs_sq_sum_{window_size}")
            fcst_sq_sum = getattr(self, f"fcst_sq_sum_{window_size}")
            obs_fcst_sum = getattr(self, f"obs_fcst_sum_{window_size}")
            num_grids = getattr(self, f"num_grids_{window_size}")

            if num_grids == 0:
                results[f"fss_window_{window_size}"] = torch.tensor(float("nan"))
                continue

            numerator = obs_sq_sum + fcst_sq_sum - 2 * obs_fcst_sum
            denominator = obs_sq_sum + fcst_sq_sum

            if denominator == 0:
                value = torch.tensor(1.0, device=obs_sq_sum.device)
            else:
                value = 1 - numerator / denominator

            results[f"fss_window_{window_size}"] = torch.clamp(value, min=0.0, max=1.0)

        return results

    def __str__(self):
        return f"FSS@{self.threshold:.3f}"


class FractionsSkillScoreMean(Metric):
    def __init__(
        self,
        threshold: float,
        window_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        reduce_zero_forecast: str = "skip",
    ):
        super().__init__()

        self.metric = FractionsSkillScore(
            threshold=threshold,
            window_sizes=window_sizes,
            reduce_zero_forecast=reduce_zero_forecast,
        )
        self.threshold = threshold

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target)

    def compute(self):
        values = self.metric.compute()
        valid = [value for value in values.values() if not torch.isnan(value)]
        if not valid:
            return torch.tensor(float("nan"))
        return torch.mean(torch.stack(valid))

    def __str__(self):
        return f"FSSMean@{self.threshold:.3f}"
