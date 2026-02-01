import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
    EarlyStopping,
    
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy


# ===================================
# ! Change to your Dataset Reader
from dataset.DemoDataset import DemoDataModule
dataset = DemoDataModule(batch_size=2)
# ===================================

# ===================================
# ! Change to your Model
from src.MyModel import MyModel
model = MyModel(input_size=128, hidden_size=64, output_size=128)
# ===================================

# * Configuration
pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
MODEL_NAME = "MyModel"

# callback
callbacks = [
    ModelCheckpoint(
        dirpath=f"checkpoints/{MODEL_NAME}",
        filename="{epoch}-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        enable_version_counter=False
    ),
    # LearningRateMonitor(logging_interval="epoch", log_weight_decay=False),
    RichProgressBar(),
    RichModelSummary(max_depth=3),
    # EarlyStopping(monitor="val/loss", patience=5, mode="min"),
]


# logger
csv_logger = CSVLogger(
    save_dir="logs", name=MODEL_NAME, flush_logs_every_n_steps=10
)


# trainer
trainer = pl.Trainer(
    max_epochs=200,
    accelerator="gpu",
    devices=[0],
    precision="32",
    log_every_n_steps=30,
    enable_model_summary=True,
    callbacks=callbacks,
    logger=[csv_logger],
    strategy=DDPStrategy(find_unused_parameters=False),
)

trainer.fit(model=model, datamodule=dataset)
