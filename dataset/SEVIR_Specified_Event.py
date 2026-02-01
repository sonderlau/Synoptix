import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from lightning.pytorch import LightningDataModule
from typing import List, Optional, Tuple
import os
from datetime import datetime
import torch.nn.functional as F


class SEVIRDataset(Dataset):
    """
    SEVIR VIL 数据集类
    读取指定气象事件类型的VIL数据，按照指定格式处理
    """

    def __init__(
            self,
            catalog_path: str,
            data_root: str,
            event_types: List[str],
            split_date: str,
            is_train: bool = True,
            input_frames: int = 12,
            output_frames: int = 12,
            stride: int = 12,
            img_size: int = 384,
    ):
        """
        初始化数据集

        Args:
            catalog_path: catalog.csv文件路径
            data_root: SEVIR数据根目录
            event_types: 要读取的事件类型列表，如['tornado', 'wind', 'hail']
            split_date: 用于划分训练/测试集的日期，格式'YYYY-MM-DD'
            is_train: 是否为训练集
            input_frames: 输入帧数 (默认6)
            output_frames: 输出帧数 (默认6)
            stride: 滑动步长 (默认12)
            img_size: 图像尺寸 (默认384)
        """
        self.catalog_path = catalog_path
        self.data_root = data_root
        self.event_types = event_types
        self.split_date = datetime.strptime(split_date, "%Y-%m-%d")
        self.is_train = is_train
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.img_size = img_size

        # 读取catalog并过滤数据
        self.catalog = pd.read_csv(catalog_path, low_memory=False)
        self.valid_samples = self._prepare_samples()

        print(f"{'训练' if is_train else '测试'}集样本数: {len(self.valid_samples)}")

    def _prepare_samples(self) -> List[dict]:
        """
        准备有效的样本列表
        """
        valid_samples = []

        if self.event_types != "":

            # 过滤指定事件类型的VIL数据
            filtered_catalog = self.catalog[
                (self.catalog["img_type"] == "vil")
                & (self.catalog["event_type"].isin(self.event_types))
                ]
        else:
            filtered_catalog = self.catalog[
                (self.catalog["img_type"] == "vil")
            ]

        for idx, row in filtered_catalog.iterrows():
            # 解析日期进行训练/测试集划分
            event_date = datetime.strptime(row["time_utc"], "%Y-%m-%d %H:%M:%S")

            if self.is_train and event_date >= self.split_date:
                continue
            if not self.is_train and event_date < self.split_date:
                continue

            # 检查该事件是否有足够的帧数

            total_needed_frames = self.input_frames + self.output_frames

            # SEVIR数据通常有49帧，我们需要检查能产生多少个有效样本
            max_frames = 49  # SEVIR标准帧数

            # 计算该事件可以产生多少个有效样本
            start_idx = 0
            while start_idx + total_needed_frames < max_frames:
                sample_info = {
                    "file_path": os.path.join(self.data_root, "data", row["file_name"]),
                    "file_index": int(row["file_index"]),
                    "img_type": row["img_type"],
                    "event_type": row["event_type"],
                    "event_id": row["id"],
                    "start_idx": start_idx,
                }
                valid_samples.append(sample_info)
                start_idx += self.stride

        return valid_samples

    def _get_frame_indices(self, start_idx: int, num_frames: int) -> List[int]:
        """
        获取跳步采样的帧索引
        从start_idx开始，每隔2帧采样一次，获取num_frames帧
        """
        indices = []
        for i in range(num_frames):
            indices.append(start_idx + i * 2)  # 每隔2帧采样
        return indices

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Returns:
            input_data: (6, 384, 384) 输入序列
            target_data: (6, 384, 384) 目标序列
        """
        sample_info = self.valid_samples[idx]

        try:
            with h5py.File(sample_info["file_path"], "r") as f:
                # 读取完整事件数据
                full_event_data = f[sample_info["img_type"]][sample_info["file_index"]]

                data = full_event_data[:, :,
                       sample_info["start_idx"]: sample_info["start_idx"] + self.input_frames + self.output_frames]

                data = self._normalize_vil_data(data)

                # 数据预处理：归一化到[0, 1]
                # input_data = self._normalize_vil_data(input_data)
                # target_data = self._normalize_vil_data(target_data)

                # 转换为torch tensor
                input_tensor = torch.from_numpy(data[:, :, :self.input_frames:2]).float().permute(2, 0, 1)
                target_tensor = torch.from_numpy(data[:, :, self.output_frames::2]).float().permute(2, 0, 1)

                return {
                    "sequence": F.interpolate(input_tensor.unsqueeze(0), size=128, mode="bilinear").squeeze(0),
                    "target": F.interpolate(target_tensor.unsqueeze(0), size=128, mode="bilinear").squeeze(0)
                }


        except Exception as e:
            print(
                f"读取数据时发生错误: {sample_info['file_path']}, 索引: {sample_info['file_index']}"
            )
            print(f"错误信息: {e}")
            # 返回零张量作为fallback
            return torch.zeros(
                self.input_frames, self.img_size, self.img_size
            ), torch.zeros(self.output_frames, self.img_size, self.img_size)

    def _normalize_vil_data(self, data: np.ndarray) -> np.ndarray:
        """
        VIL数据归一化
        SEVIR VIL数据范围通常是0-255，归一化到[0, 1]
        """
        # 确保数据在合理范围内
        data = np.clip(data, 0, 255)
        data[data == 255] = 0
        # 归一化到[0, 1]
        return data / 255.0


class SEVIRDataModule(LightningDataModule):
    """
    SEVIR 数据模块 (PyTorch Lightning)
    """

    def __init__(
            self,
            catalog_path: str,
            data_root: str,
            event_types: List[str],
            split_date: str,
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            seed: int = 42,
            input_frames: int = 12,
            output_frames: int = 12,
            stride: int = 12,
            img_size: int = 384,
    ):
        """
        初始化DataModule

        Args:
            catalog_path: catalog.csv文件路径
            data_root: SEVIR数据根目录
            event_types: 要读取的事件类型列表
            split_date: 用于划分训练/测试集的日期
            batch_size: 批次大小
            num_workers: 数据加载器工作线程数
            pin_memory: 是否使用pin_memory
            seed: 随机种子
            input_frames: 输入帧数
            output_frames: 输出帧数
            stride: 滑动步长
            img_size: 图像尺寸
        """
        super().__init__()
        self.save_hyperparameters()

        self.catalog_path = catalog_path
        self.data_root = data_root
        self.event_types = event_types
        self.split_date = split_date
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.img_size = img_size

        # 设置随机种子
        pl.seed_everything(self.seed)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        设置数据集
        """

        val_ratio = 0.1

        full_train_dataset = SEVIRDataset(
            catalog_path=self.catalog_path,
            data_root=self.data_root,
            event_types=self.event_types,
            split_date=self.split_date,
            is_train=True,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
            stride=self.stride,
            img_size=self.img_size,
        )

        total_size = len(full_train_dataset)
        val_size = int(total_size * val_ratio)

        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        print(f"Train size: {len(self.train_dataset)}; val size: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# 使用示例
if __name__ == "__main__":
    # 创建数据模块
    datamodule = SEVIRDataModule(
        catalog_path="/data3/SEVIR/CATALOG.csv",
        data_root="/data3/SEVIR",
        event_types=["Flash Flood"],  # 指定要读取的事件类型
        split_date="2019-06-01",  # 2019年之前为训练集，之后为测试集
        batch_size=16,
        num_workers=4,
        seed=42,
    )

    # 设置数据集
    datamodule.setup()

    # 获取训练数据加载器
    train_loader = datamodule.train_dataloader()

    # 测试数据加载
    print("测试数据加载...")
    for batch_idx, data in enumerate(train_loader):
        inputs = data["sequence"]
        targets = data["target"]
        print(f"Batch {batch_idx}:")
        print(f"  输入形状: {inputs.shape}")  # 应该是 (batch_size, 6, 384, 384)
        print(f"  目标形状: {targets.shape}")  # 应该是 (batch_size, 6, 384, 384)
        print(f"  输入数据范围: {inputs.min():.3f} - {inputs.max():.3f}")
        print(f"  目标数据范围: {targets.min():.3f} - {targets.max():.3f}")

        if batch_idx >= 2:  # 只测试前3个批次
            break

    print("数据模块测试完成！")
