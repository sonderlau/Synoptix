import torch

def zncc(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8):
    """
    Zero-mean Normalized Cross-Correlation Loss

    :param pred: prediction, shape: (*, H, W)
    :param truth: ground truth, shape: (*, H, W), same as pred
    :param eps: small value to avoid division by zero
    :return: the mean ZNCC loss, [-1, 1]
    """

    assert pred.shape == truth.shape, "Prediction and truth must have the same shape"

    spatial_dims = [-2, -1]

    # Zero-mean step
    mean_pred = pred.mean(dim=spatial_dims, keepdim=True)
    mean_truth = truth.mean(dim=spatial_dims, keepdim=True)

    # centering
    pred_centered = pred - mean_pred
    truth_centered = truth - mean_truth

    # sum((I - mean_I) * (T - mean_T))
    numerator = (pred_centered * truth_centered).sum(dim=spatial_dims)

    # sqrt(sum((I - mean_I)^2)) * sqrt(sum((T - mean_T)^2))
    std_pred = torch.sqrt((pred_centered ** 2).sum(dim=spatial_dims) + eps)
    std_truth = torch.sqrt((truth_centered ** 2).sum(dim=spatial_dims) + eps)

    denominator = std_pred * std_truth

    zncc_score = numerator / (denominator + eps)

    return zncc_score.mean()