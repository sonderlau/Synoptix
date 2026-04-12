import torch


def flatten_spatiotemporal(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() == 4:
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        target = target.reshape(-1, target.shape[-2], target.shape[-1])
    return pred, target


def safe_tensor(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float, device=reference.device)
