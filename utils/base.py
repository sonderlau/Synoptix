from enum import Enum
from torch import Tensor

class PredictionType:
    pred: Tensor
    target: Tensor