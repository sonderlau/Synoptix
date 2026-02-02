from typing import Literal

import torch
import torch.nn as nn

from lightning.fabric.utilities.throughput import measure_flops
from torchinfo import summary


# * Starts here
# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))


model = SimpleCNN()
# 用meta张量模拟输入（无内存占用），验证输出形状
meta_input = torch.randn(1, 3, 32, 32, device="meta")

# =========================

def convert_flops_units(flops: int, decimal_places: int = 2) -> str:
    """
    将FLOPS整数转换为带单位的可读字符串（自动选择M/G/T等单位）

    参数：
        flops: 计算得到的原始FLOPS数值（int）
        decimal_places: 保留的小数位数，默认2位

    返回：
        格式化字符串，如"8.20 M"、"12.50 G"
    """
    # 定义单位和对应的换算系数
    units = [
        ("", 1),  # 原始单位（无）
        ("K", 1e3),  # 千
        ("M", 1e6),  # 百万
        ("G", 1e9),  # 十亿
        ("T", 1e12)  # 万亿
    ]

    # 从大到小遍历，找到最合适的单位
    for unit, coeff in reversed(units):
        if flops >= coeff or coeff == 1:  # 至少匹配到原始单位
            converted = flops / coeff
            return f"{converted:.{decimal_places}f} {unit}"

    # 兜底（数值为0的情况）
    return f"0.00 "

fwd_flops = measure_flops(model, lambda: model(meta_input))

summary(model, input_data=meta_input)


print(f"FLOPs: {convert_flops_units(fwd_flops, decimal_places=6)}")