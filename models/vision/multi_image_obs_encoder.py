from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torchvision

from models.common.module_attr_mixin import ModuleAttrMixin
from models.vision.crop_randomizer import CropRandomizer
from utils.common.pytorch_util import replace_submodules


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model: nn.Module,
        resize_shape=None,
        crop_shape=None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
    ) -> None:
        super().__init__()
        self.shape_meta = dict(shape_meta)
        self.share_rgb_model = bool(share_rgb_model)
        self.rgb_keys: list[str] = []
        self.low_dim_keys: list[str] = []
        self.key_shape_map: dict[str, tuple[int, ...]] = {}
        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()

        if self.share_rgb_model:
            shared_rgb_channels = next(
                int(attr["shape"][0])
                for attr in self.shape_meta["obs"].values()
                if attr.get("type", "low_dim") == "rgb"
            )
            self.key_model_map["rgb"] = self._prepare_rgb_model(
                rgb_model,
                use_group_norm=use_group_norm,
                input_channels=shared_rgb_channels,
            )

        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            self.key_shape_map[key] = shape
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
                if not self.share_rgb_model:
                    self.key_model_map[key] = self._prepare_rgb_model(
                        copy.deepcopy(rgb_model),
                        use_group_norm=use_group_norm,
                        input_channels=shape[0],
                    )
                self.key_transform_map[key] = self._build_transform(
                    key=key,
                    shape=shape,
                    resize_shape=resize_shape,
                    crop_shape=crop_shape,
                    random_crop=random_crop,
                    imagenet_norm=imagenet_norm,
                )
                continue
            assert obs_type == "low_dim", f"Unsupported obs type '{obs_type}'."
            self.low_dim_keys.append(key)

        self.rgb_keys = sorted(self.rgb_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)

    def _prepare_rgb_model(
        self,
        rgb_model: nn.Module,
        use_group_norm: bool,
        input_channels: int,
    ) -> nn.Module:
        conv1 = getattr(rgb_model, "conv1", None)
        if conv1 is not None and hasattr(conv1, "in_channels"):
            if int(conv1.in_channels) != int(input_channels):
                replacement = nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=conv1.out_channels,
                    kernel_size=conv1.kernel_size,
                    stride=conv1.stride,
                    padding=conv1.padding,
                    bias=conv1.bias is not None,
                )
                with torch.no_grad():
                    if int(conv1.in_channels) == 3 and input_channels % 3 == 0:
                        repeats = input_channels // 3
                        repeated = conv1.weight.repeat(1, repeats, 1, 1) / float(
                            repeats
                        )
                        replacement.weight.copy_(repeated)
                    else:
                        replacement.weight.copy_(
                            conv1.weight.mean(dim=1, keepdim=True).repeat(
                                1, input_channels, 1, 1
                            )
                        )
                    if conv1.bias is not None:
                        replacement.bias.copy_(conv1.bias)
                rgb_model.conv1 = replacement
        if not use_group_norm:
            return rgb_model
        return replace_submodules(
            root_module=rgb_model,
            predicate=lambda module: isinstance(module, nn.BatchNorm2d),
            func=lambda module: nn.GroupNorm(
                num_groups=max(1, module.num_features // 16),
                num_channels=module.num_features,
            ),
        )

    def _build_transform(
        self,
        key: str,
        shape: tuple[int, ...],
        resize_shape,
        crop_shape,
        random_crop: bool,
        imagenet_norm: bool,
    ) -> nn.Module:
        input_shape = shape
        transforms: list[nn.Module] = []
        if resize_shape is not None:
            if isinstance(resize_shape, dict):
                height, width = resize_shape[key]
            else:
                height, width = resize_shape
            transforms.append(torchvision.transforms.Resize((height, width)))
            input_shape = (shape[0], height, width)

        if crop_shape is not None:
            if isinstance(crop_shape, dict):
                crop_height, crop_width = crop_shape[key]
            else:
                crop_height, crop_width = crop_shape
            if random_crop:
                transforms.append(
                    CropRandomizer(
                        input_shape=input_shape,
                        crop_height=crop_height,
                        crop_width=crop_width,
                        num_crops=1,
                        pos_enc=False,
                    )
                )
            else:
                transforms.append(
                    torchvision.transforms.CenterCrop((crop_height, crop_width))
                )

        if imagenet_norm:
            channels = input_shape[0]
            assert channels % 3 == 0, (
                "imagenet_norm requires rgb channels grouped in triples, "
                f"got {channels} channels for key '{key}'."
            )
            mean = [0.485, 0.456, 0.406] * (channels // 3)
            std = [0.229, 0.224, 0.225] * (channels // 3)
            transforms.append(torchvision.transforms.Normalize(mean=mean, std=std))

        if not transforms:
            return nn.Identity()
        return nn.Sequential(*transforms)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        features = []

        if self.share_rgb_model:
            rgb_inputs = []
            for key in self.rgb_keys:
                image = obs_dict[key]
                if batch_size is None:
                    batch_size = image.shape[0]
                else:
                    assert batch_size == image.shape[0]
                assert image.shape[1:] == self.key_shape_map[key]
                rgb_inputs.append(self.key_transform_map[key](image))
            if rgb_inputs:
                stacked = torch.cat(rgb_inputs, dim=0)
                rgb_features = self.key_model_map["rgb"](stacked)
                rgb_features = rgb_features.reshape(
                    -1, batch_size, *rgb_features.shape[1:]
                )
                rgb_features = torch.movedim(rgb_features, 0, 1)
                features.append(rgb_features.reshape(batch_size, -1))
        else:
            for key in self.rgb_keys:
                image = obs_dict[key]
                if batch_size is None:
                    batch_size = image.shape[0]
                else:
                    assert batch_size == image.shape[0]
                assert image.shape[1:] == self.key_shape_map[key]
                transformed = self.key_transform_map[key](image)
                features.append(self.key_model_map[key](transformed))

        for key in self.low_dim_keys:
            value = obs_dict[key]
            if batch_size is None:
                batch_size = value.shape[0]
            else:
                assert batch_size == value.shape[0]
            assert value.shape[1:] == self.key_shape_map[key]
            features.append(value)

        assert features, "Observation encoder received no features."
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs = {}
        for key, attr in self.shape_meta["obs"].items():
            shape = tuple(attr["shape"])
            example_obs[key] = torch.zeros(
                (1,) + shape, dtype=self.dtype, device=self.device
            )
        output = self.forward(example_obs)
        return output.shape[1:]
