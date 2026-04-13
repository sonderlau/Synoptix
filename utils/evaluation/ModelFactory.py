from __future__ import annotations

from typing import Any

import torch

from utils.evaluation.base import ModelStrategy, summary_print

try:
    from src.MyModel import MyModel
except ImportError as exc:
    raise ImportError(f"Failed to import model definitions: {exc}") from exc


STRATEGY_REGISTRY: dict[str, type[ModelStrategy]] = {}


def register_strategy(name: str):
    def decorator(cls: type[ModelStrategy]):
        STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


@register_strategy("MyModel")
class MyModelStrategy(ModelStrategy):
    def build_model(self) -> torch.nn.Module:
        return MyModel(input_size=128, hidden_size=64, output_size=128)


class ModelFactory:
    """Creates inference-ready models from registered strategies."""

    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        device: torch.device,
    ):
        if model_name not in STRATEGY_REGISTRY:
            available = ", ".join(sorted(STRATEGY_REGISTRY))
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        self.model_name = model_name
        self.strategy = STRATEGY_REGISTRY[model_name]()
        self.device = device
        self.model = self.strategy.load_checkpoint(ckpt_path, map_location="cpu")
        self.model.to(device)
        self.model.eval()
        self._summary_result: str | None = None

    def infer(
        self,
        data: torch.Tensor,
        model: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        inference_model = model or self.model
        return self.strategy.inference(inference_model, data)

    def cache_summary(self, sample: torch.Tensor) -> None:
        if self._summary_result is not None:
            return

        try:
            input_data = self.strategy.preprocess(sample.detach().clone())
            self._summary_result = summary_print(
                model=self.model,
                strategy=self.strategy,
                input_data=input_data,
            )
            self.model.eval()
        except Exception as exc:
            self._summary_result = f"Summary calculation failed: {exc}"

    def print_summary(self, console: Any) -> None:
        if self._summary_result:
            console.print(
                f"\n[bold cyan]Model Statistics: {self.model_name}[/bold cyan]"
            )
            console.print(self._summary_result)
            return

        console.print(
            f"[yellow]No summary info available for {self.model_name}[/yellow]"
        )
