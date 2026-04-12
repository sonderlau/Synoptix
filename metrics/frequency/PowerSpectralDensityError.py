import torch
from torchmetrics import Metric


class PowerSpectraDensityError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psd_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        *_, height, width = pred.shape

        pred = pred.float()
        target = target.float()

        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))

        power_pred = torch.abs(fft_pred) ** 2 / (height * width)
        power_target = torch.abs(fft_target) ** 2 / (height * width)

        self.psd_error += torch.nn.functional.mse_loss(power_pred, power_target)
        self.count += 1

    def compute(self):
        return self.psd_error / self.count

    def __str__(self):
        return "PSD"
