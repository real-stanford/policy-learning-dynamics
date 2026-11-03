"""Thin wrapper over diffusers' learning-rate schedules."""

from __future__ import annotations

from typing import Optional

from diffusers.optimization import (
    TYPE_TO_SCHEDULER_FUNCTION,
    Optimizer,
    SchedulerType,
)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    scheduler_type = SchedulerType(name)
    schedule_fn = TYPE_TO_SCHEDULER_FUNCTION[scheduler_type]
    if scheduler_type == SchedulerType.CONSTANT:
        return schedule_fn(optimizer, **kwargs)
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
    if scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_fn(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
    return schedule_fn(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
