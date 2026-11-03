from __future__ import annotations

import torch
import torch._inductor.config as inductor_config


def compile_module(module, cfg, name: str):
    if not bool(cfg.enabled):
        return module
    assert hasattr(torch, "compile"), (
        "torch.compile is not available in this PyTorch build."
    )

    inductor_config.triton.cudagraphs = bool(cfg.cudagraphs)
    if hasattr(inductor_config.triton, "cudagraph_trees"):
        inductor_config.triton.cudagraph_trees = bool(cfg.cudagraphs)

    mode = str(cfg.mode)
    if not bool(cfg.cudagraphs) and mode == "reduce-overhead":
        mode = "default"

    return torch.compile(
        module,
        mode=mode,
        fullgraph=bool(cfg.fullgraph),
        dynamic=bool(cfg.dynamic),
    )


def unwrap_compiled_module(module):
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _normalize_compiled_state_key(key: str) -> str:
    clean_key = str(key)
    while "._orig_mod." in clean_key:
        clean_key = clean_key.replace("._orig_mod.", ".")
    if clean_key.startswith("_orig_mod."):
        clean_key = clean_key[len("_orig_mod.") :]
    return clean_key


def sanitize_compiled_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    sanitized = {}
    for key, value in state_dict.items():
        clean_key = _normalize_compiled_state_key(str(key))
        assert clean_key not in sanitized, (
            f"Duplicate state dict key after compile sanitization: '{clean_key}'."
        )
        sanitized[clean_key] = value
    return sanitized


def remap_state_dict_for_module(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    sanitized_state = sanitize_compiled_state_dict(state_dict)
    normalized_target_keys = {}
    for actual_key in module.state_dict().keys():
        normalized_key = _normalize_compiled_state_key(actual_key)
        assert normalized_key not in normalized_target_keys, (
            "Duplicate normalized target key while remapping compiled state dict: "
            f"'{normalized_key}'."
        )
        normalized_target_keys[normalized_key] = actual_key

    source_keys = set(sanitized_state.keys())
    target_keys = set(normalized_target_keys.keys())
    assert source_keys == target_keys, (
        "Checkpoint state keys do not match target module keys after compile-aware "
        f"normalization. Missing={sorted(target_keys - source_keys)[:5]} "
        f"Unexpected={sorted(source_keys - target_keys)[:5]} "
        f"MissingCount={len(target_keys - source_keys)} "
        f"UnexpectedCount={len(source_keys - target_keys)}"
    )
    return {
        normalized_target_keys[normalized_key]: tensor
        for normalized_key, tensor in sanitized_state.items()
    }


def load_compiled_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    remapped_state = remap_state_dict_for_module(module, state_dict)
    load_result = module.load_state_dict(remapped_state)
    assert not load_result.missing_keys, (
        f"Missing keys after compile-aware load: {load_result.missing_keys}."
    )
    assert not load_result.unexpected_keys, (
        f"Unexpected keys after compile-aware load: {load_result.unexpected_keys}."
    )


def compile_policy_modules(policy, cfg, prefix: str):
    """torch.compile the policy, or only its U-Net when a generated-sample mixer is
    attached: the mixer's mix_batch runs eager generation inside compute_loss (it is
    @torch._dynamo.disable'd), which must stay outside the compiled, DDP-wrapped forward."""
    if getattr(policy, "oa_joint_mixer", None) is not None:
        policy.model = compile_module(policy.model, cfg, f"{prefix}_unet")
        return policy
    return compile_module(policy, cfg, f"{prefix}_policy")
