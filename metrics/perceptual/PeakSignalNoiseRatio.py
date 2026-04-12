import torch


def PSNR(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"), device=img1.device)
    return 10 * torch.log10((data_range ** 2) / mse)
