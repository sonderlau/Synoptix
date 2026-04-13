import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.accelerators import CUDAAccelerator
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor, set_float32_matmul_precision
from torchmetrics import MetricCollection

from dataset.DataReader import HDF5DataModule
from metrics import CSI, FAR, POD, RMSE, SSIM, CSIMean, FSSMean, HSSMean
from utils.evaluation.ModelFactory import ModelFactory
from utils.statistic import print_result

set_float32_matmul_precision("high")
seed_everything(42, workers=True)

console = Console()

DTYPE = torch.float32
LEAD_TIMES: Tuple[int, ...] = (10, 20, 30, 40, 50, 60)
VALUE_SCALE = 255.0
THRESHOLDS: Tuple[int, ...] = (181, 219)
FSS_WINDOW_SIZES: Tuple[int, ...] = (3,)
MAX_INFERENCE_DEVICES: int | None = None

CHECKPOINTS = {"MyModel": "checkpoints/MyModel/last.ckpt"}


@dataclass(frozen=True)
class InferenceRuntime:
    primary_device: torch.device
    device_ids: Tuple[int, ...]
    use_data_parallel: bool


def resolve_runtime(max_devices: int | None = None) -> InferenceRuntime:
    if not torch.cuda.is_available():
        return InferenceRuntime(
            primary_device=torch.device("cpu"),
            device_ids=(),
            use_data_parallel=False,
        )

    device_count = CUDAAccelerator.auto_device_count()
    if max_devices is not None:
        device_count = min(device_count, max_devices)

    device_ids = tuple(range(max(1, device_count)))
    return InferenceRuntime(
        primary_device=torch.device(f"cuda:{device_ids[0]}"),
        device_ids=device_ids,
        use_data_parallel=len(device_ids) > 1,
    )


RUNTIME = resolve_runtime(MAX_INFERENCE_DEVICES)

METRICS_PROTOTYPE = [
    RMSE(),
    SSIM(),
    HSSMean(thresholds=THRESHOLDS),
    CSIMean(thresholds=THRESHOLDS),
    CSI(threshold=181),
    CSI(threshold=219),
    POD(threshold=181),
    POD(threshold=219),
    FAR(threshold=181),
    FAR(threshold=219),
    FSSMean(threshold=181, window_sizes=FSS_WINDOW_SIZES),
    FSSMean(threshold=219, window_sizes=FSS_WINDOW_SIZES),
]

METRIC_NAMES = [str(metric) for metric in METRICS_PROTOTYPE]


def setup_metrics(
    lead_times: Tuple[int, ...],
    prototype: Iterable,
) -> Dict[str, MetricCollection]:
    metric_collections: Dict[str, MetricCollection] = {}
    for lead_time in lead_times:
        metrics = {str(metric): copy.deepcopy(metric) for metric in prototype}
        metric_collections[f"lead_{lead_time}"] = MetricCollection(metrics).to(
            RUNTIME.primary_device
        )
    return metric_collections


def get_batch_tensors(batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    if "sequence" not in batch or "target" not in batch:
        raise KeyError("Each batch must contain 'sequence' and 'target'.")

    sequence = batch["sequence"].to(
        device=RUNTIME.primary_device,
        dtype=DTYPE,
        non_blocking=True,
    )
    target = batch["target"].to(
        device=RUNTIME.primary_device,
        dtype=DTYPE,
        non_blocking=True,
    )

    return sequence.contiguous(), target.contiguous()


def validate_prediction(pred: Tensor, target: Tensor) -> None:
    if pred.ndim < 2 or target.ndim < 2:
        raise ValueError("Prediction and target must include a time dimension.")

    if pred.size(1) < len(LEAD_TIMES):
        raise ValueError(
            f"Prediction only has {pred.size(1)} lead steps, "
            f"but {len(LEAD_TIMES)} are required."
        )

    if target.size(1) < len(LEAD_TIMES):
        raise ValueError(
            f"Target only has {target.size(1)} lead steps, "
            f"but {len(LEAD_TIMES)} are required."
        )


def evaluate_model(model_name: str, ckpt_path: str, val_loader) -> None:
    console.rule(f"[bold yellow]Evaluating Model: {model_name}")

    model_factory = ModelFactory(
        model_name=model_name,
        ckpt_path=ckpt_path,
        device=RUNTIME.primary_device,
    )
    inference_model: torch.nn.Module = model_factory.model
    if RUNTIME.use_data_parallel:
        inference_model = torch.nn.DataParallel(
            model_factory.model,
            device_ids=list(RUNTIME.device_ids),
            output_device=RUNTIME.device_ids[0],
        )

    metrics_per_lead = setup_metrics(LEAD_TIMES, METRICS_PROTOTYPE)

    with torch.inference_mode():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Inference {model_name}...",
                total=len(val_loader),
            )

            for batch in val_loader:
                sequence, target = get_batch_tensors(batch)
                model_factory.cache_summary(sequence[:1])
                pred = model_factory.infer(sequence, model=inference_model)
                validate_prediction(pred, target)

                for index, lead_time in enumerate(LEAD_TIMES):
                    lead_pred = pred[:, index : index + 1, ...]
                    lead_target = target[:, index : index + 1, ...]
                    metrics_per_lead[f"lead_{lead_time}"].update(
                        lead_pred * VALUE_SCALE,
                        lead_target * VALUE_SCALE,
                    )

                progress.advance(task)

    table = print_result(
        model_name=model_name,
        metric_collection=metrics_per_lead,
        lead_times=LEAD_TIMES,
        metric_order=METRIC_NAMES,
    )
    console.print(table)

    model_factory.print_summary(console)
    console.print()

    del inference_model
    del model_factory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    console.print(
        f"[bold cyan]Inference Runtime[/bold cyan] "
        f"device={RUNTIME.primary_device} "
        f"device_ids={list(RUNTIME.device_ids)} "
        f"data_parallel={RUNTIME.use_data_parallel}"
    )

    dataset = HDF5DataModule("/data2/StormWave/SEVIR", batch_size=25, num_workers=128)
    dataset.setup("fit")
    val_loader = dataset.val_dataloader()

    for model_name, ckpt_path in CHECKPOINTS.items():
        evaluate_model(model_name, ckpt_path, val_loader)


if __name__ == "__main__":
    main()
