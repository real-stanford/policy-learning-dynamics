from __future__ import annotations

import re
import sys

import torch
from omegaconf import OmegaConf


def sanitize_exp_suffix(raw_value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_value).strip())
    cleaned = cleaned.strip("_")
    assert cleaned, "Experiment suffix must contain at least one valid character."
    return cleaned


def format_run_name(datetime_stamp: str, exp_suffix: str | None) -> str:
    if exp_suffix in (None, "", "null", "None"):
        return str(datetime_stamp)
    return f"{datetime_stamp}_{sanitize_exp_suffix(exp_suffix)}"


def register_training_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("run_name", format_run_name, replace=True)


def configure_torch_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def consume_training_cli_args() -> None:
    """Translate the two convenience flags into Hydra overrides before Hydra parses argv."""
    argv = list(sys.argv)
    passthrough = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--exp_suffix":
            assert index + 1 < len(argv), "--exp_suffix requires a value."
            passthrough.append(
                f"runtime.exp_suffix={sanitize_exp_suffix(argv[index + 1])}"
            )
            index += 2
            continue
        if arg in {"--smoke", "--smoke-test"}:
            passthrough.append("smoke.enabled=true")
            index += 1
            continue
        passthrough.append(arg)
        index += 1
    sys.argv[:] = passthrough


def apply_smoke_overrides(cfg) -> None:
    if not bool(cfg.smoke.enabled):
        return
    train_cfg = cfg.training.training
    train_cfg.num_epochs = int(cfg.smoke.num_epochs)
    train_cfg.save_every = 1
    train_cfg.val_every = 1
    train_cfg.sample_every = 1
    train_cfg.max_train_steps = int(cfg.smoke.max_train_steps)
    train_cfg.max_val_steps = int(cfg.smoke.max_val_steps)
