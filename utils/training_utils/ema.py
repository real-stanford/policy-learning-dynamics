from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.nn.modules.batchnorm import _BatchNorm


@dataclass
class EMAConfig:
    update_after_step: int
    inv_gamma: float
    power: float
    min_value: float
    max_value: float


def _iter_unwrapped_modules(
    module: torch.nn.Module,
    prefix: str = "",
) -> Iterator[tuple[str, torch.nn.Module]]:
    compiled_module = getattr(module, "_orig_mod", None)
    if compiled_module is not None:
        assert isinstance(compiled_module, torch.nn.Module), (
            f"Expected compiled module to wrap torch.nn.Module, got {type(compiled_module)}."
        )
        yield from _iter_unwrapped_modules(compiled_module, prefix)
        return

    yield prefix, module
    for child_name, child in module.named_children():
        child_prefix = child_name if not prefix else f"{prefix}.{child_name}"
        yield from _iter_unwrapped_modules(child, child_prefix)


def _format_missing_keys(
    expected: set[str],
    actual: set[str],
    label: str,
) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return (
        f"{label} mismatch."
        f" Missing={missing[:5]}"
        f" Unexpected={unexpected[:5]}"
        f" MissingCount={len(missing)}"
        f" UnexpectedCount={len(unexpected)}"
    )


class _EMAModel:
    def __init__(
        self,
        model,
        update_after_step: int,
        inv_gamma: float,
        power: float,
        min_value: float,
        max_value: float,
    ) -> None:
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model) -> None:
        self.decay = self.get_decay(self.optimization_step)
        source_modules = dict(_iter_unwrapped_modules(new_model))
        averaged_modules = dict(_iter_unwrapped_modules(self.averaged_model))
        assert source_modules.keys() == averaged_modules.keys(), _format_missing_keys(
            set(averaged_modules),
            set(source_modules),
            "EMA module tree",
        )

        for module_name, averaged_module in averaged_modules.items():
            source_module = source_modules[module_name]
            assert type(source_module) is type(averaged_module), (
                f"EMA module type mismatch for '{module_name}':"
                f" source={type(source_module)} averaged={type(averaged_module)}."
            )

            source_params = dict(source_module.named_parameters(recurse=False))
            averaged_params = dict(averaged_module.named_parameters(recurse=False))
            assert source_params.keys() == averaged_params.keys(), _format_missing_keys(
                set(averaged_params),
                set(source_params),
                f"EMA parameter set for module '{module_name}'",
            )
            for param_name, averaged_param in averaged_params.items():
                source_param = source_params[param_name]
                assert source_param.shape == averaged_param.shape, (
                    f"EMA parameter shape mismatch for '{module_name}.{param_name}':"
                    f" source={tuple(source_param.shape)}"
                    f" averaged={tuple(averaged_param.shape)}."
                )
                if (
                    isinstance(source_module, _BatchNorm)
                    or not source_param.requires_grad
                ):
                    averaged_param.copy_(source_param.to(dtype=averaged_param.dtype))
                    continue
                averaged_param.mul_(self.decay)
                averaged_param.add_(
                    source_param.to(dtype=averaged_param.dtype),
                    alpha=1 - self.decay,
                )

            source_buffers = dict(source_module.named_buffers(recurse=False))
            averaged_buffers = dict(averaged_module.named_buffers(recurse=False))
            assert source_buffers.keys() == averaged_buffers.keys(), (
                _format_missing_keys(
                    set(averaged_buffers),
                    set(source_buffers),
                    f"EMA buffer set for module '{module_name}'",
                )
            )
            for buffer_name, averaged_buffer in averaged_buffers.items():
                source_buffer = source_buffers[buffer_name]
                assert source_buffer.shape == averaged_buffer.shape, (
                    f"EMA buffer shape mismatch for '{module_name}.{buffer_name}':"
                    f" source={tuple(source_buffer.shape)}"
                    f" averaged={tuple(averaged_buffer.shape)}."
                )
                if averaged_buffer.is_floating_point():
                    averaged_buffer.copy_(source_buffer.to(dtype=averaged_buffer.dtype))
                    continue
                averaged_buffer.copy_(source_buffer)
        self.optimization_step += 1


class EMAHandler:
    def __init__(self, model, cfg: EMAConfig) -> None:
        self.ema_model = copy.deepcopy(model)
        self.updater = _EMAModel(
            model=self.ema_model,
            update_after_step=cfg.update_after_step,
            inv_gamma=cfg.inv_gamma,
            power=cfg.power,
            min_value=cfg.min_value,
            max_value=cfg.max_value,
        )

    def step(self, model) -> None:
        self.updater.step(model)
