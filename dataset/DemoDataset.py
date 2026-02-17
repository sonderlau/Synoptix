import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

# 自定义数据集，返回固定的 Tensor
class DemoDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.labels = torch.tensor([0, 1, 0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 定义 LightningDataModule
class DemoDataModule(LightningDataModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = DemoDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

# # 使用示例
# if __name__ == "__main__":
#     demo_dm = DemoDataModule(batch_size=2)
#     demo_dm.setup()
#     for batch in demo_dm.train_dataloader():
#         print(batch)