from typing import Tuple
import torch
from torchmetrics import Metric


class FractionsSkillScore(Metric):
    """
    Fractions Skill Score (FSS) metric for evaluating spatial forecasts.

    The FSS measures the skill of a forecast compared to a reference forecast
    by comparing the fraction of grid boxes exceeding a threshold within
    neighborhoods of various sizes.

    Args:
        threshold: Threshold value for defining events
        window_sizes: Tuple of neighborhood window sizes to evaluate
        reduce_zero_forecast: How to handle cases where forecast fraction is 0
    """

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

        # Add states for each window size
        for i, ws in enumerate(window_sizes):
            self.add_state(
                f"obs_sq_sum_{ws}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"fcst_sq_sum_{ws}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"obs_fcst_sum_{ws}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"num_grids_{ws}", default=torch.tensor(0), dist_reduce_fx="sum"
            )

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

        # Convert to binary fields based on threshold
        obs_binary = (target >= self.threshold).float()
        fcst_binary = (pred >= self.threshold).float()

        # Process each window size
        for ws in self.window_sizes:
            # Calculate fractions in sliding windows
            obs_fraction = self._compute_fractions(obs_binary, ws)
            fcst_fraction = self._compute_fractions(fcst_binary, ws)

            # Skip samples where forecast fraction is zero if specified
            if self.reduce_zero_forecast == "skip":
                mask = fcst_fraction != 0
                obs_fraction = obs_fraction[mask]
                fcst_fraction = fcst_fraction[mask]

            # Compute sums for FSS calculation
            obs_sq_sum = torch.sum(obs_fraction**2)
            fcst_sq_sum = torch.sum(fcst_fraction**2)
            obs_fcst_sum = torch.sum(obs_fraction * fcst_fraction)
            num_grids = torch.tensor(obs_fraction.numel(), dtype=torch.float)

            # Accumulate values
            setattr(
                self, f"obs_sq_sum_{ws}", getattr(self, f"obs_sq_sum_{ws}") + obs_sq_sum
            )
            setattr(
                self,
                f"fcst_sq_sum_{ws}",
                getattr(self, f"fcst_sq_sum_{ws}") + fcst_sq_sum,
            )
            setattr(
                self,
                f"obs_fcst_sum_{ws}",
                getattr(self, f"obs_fcst_sum_{ws}") + obs_fcst_sum,
            )
            setattr(
                self, f"num_grids_{ws}", getattr(self, f"num_grids_{ws}") + num_grids
            )

    def _compute_fractions(self, binary_field: torch.Tensor, window_size: int):
        """
        Compute fractions of positive events in sliding windows.

        Args:
            binary_field: Binary field of shape (batch, height, width)
            window_size: Size of the sliding window

        Returns:
            Fractions tensor of shape (batch, new_height, new_width)
        """
        if window_size == 1:
            return binary_field

        # Use unfold to create sliding windows
        unfolded = binary_field.unfold(1, window_size, 1).unfold(2, window_size, 1)
        # Compute mean across the window dimensions
        fractions = unfolded.mean(dim=(-2, -1))
        return fractions

    def compute(self):
        """
        Compute the FSS for each window size.

        Returns:
            Dictionary mapping window size to FSS value
        """
        results = {}

        for ws in self.window_sizes:
            obs_sq_sum = getattr(self, f"obs_sq_sum_{ws}")
            fcst_sq_sum = getattr(self, f"fcst_sq_sum_{ws}")
            obs_fcst_sum = getattr(self, f"obs_fcst_sum_{ws}")
            num_grids = getattr(self, f"num_grids_{ws}")

            if num_grids == 0:
                results[f"fss_window_{ws}"] = torch.tensor(float("nan"))
                continue

            # Calculate MSE for observation and forecast
            obs_mse = obs_sq_sum / num_grids
            fcst_mse = fcst_sq_sum / num_grids
            mixed_term = 2 * obs_fcst_sum / num_grids

            mse = obs_mse + fcst_mse - mixed_term

            # Calculate reference MSE (when forecast is climatology, i.e., obs mean)
            total_obs = obs_sq_sum.sqrt()  # Approximation
            ref_mse = 2 * obs_mse  # Simplified reference

            # Calculate FSS
            if ref_mse == 0:
                fss = torch.tensor(1.0 if mse == 0 else 0.0)
            else:
                fss = 1 - (mse / ref_mse)

            results[f"fss_window_{ws}"] = torch.clamp(fss, min=0.0, max=1.0)

        return results


class FractionsSkillScoreMean(Metric):
    """
    Mean Fractions Skill Score across multiple window sizes.
    """

    def __init__(
        self,
        threshold: float,
        window_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        reduce_zero_forecast: str = "skip",
    ):
        super().__init__()

        self.fss_metric = FractionsSkillScore(
            threshold, window_sizes, reduce_zero_forecast
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.fss_metric.update(pred, target)

    def compute(self):
        fss_values = self.fss_metric.compute()
        # Extract just the FSS values and compute mean
        fss_vals = torch.tensor(
            [val for val in fss_values.values() if not torch.isnan(val)]
        )
        if len(fss_vals) == 0:
            return torch.tensor(float("nan"))
        return torch.mean(fss_vals)
