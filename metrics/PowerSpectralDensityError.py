import torch
from torchmetrics import Metric


class PowerSpectraDensityError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psd_error", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):

        *_, h, w = pred.shape

        pred = pred.float()
        target = target.float()

        fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        fft_target = torch.fft.fft2(target, dim=(-2, -1))

        power_pred = torch.abs(fft_pred) ** 2 / (h * w)
        power_target = torch.abs(fft_target) ** 2 / (h * w)

        psd_value =  torch.nn.functional.mse_loss(power_pred, power_target)

        self.psd_error += psd_value
        self.count += 1

    def compute(self):
        return self.psd_error / self.count
