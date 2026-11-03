from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf

from models.common import tensor_util as tu


class CropRandomizer(nn.Module):
    def __init__(
        self,
        input_shape,
        crop_height: int,
        crop_width: int,
        num_crops: int = 1,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()
        assert len(input_shape) == 3, f"Expected CHW input shape, got {input_shape}."
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = tuple(input_shape)
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.num_crops = int(num_crops)
        self.pos_enc = bool(pos_enc)

    def forward_in(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.ndim >= 3, f"Expected tensor with CHW suffix, got {inputs.shape}."
        if self.training:
            outputs, _ = sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            return tu.join_dimensions(outputs, 0, 1)

        outputs = ttf.center_crop(
            img=inputs,
            output_size=(self.crop_height, self.crop_width),
        )
        if self.num_crops > 1:
            batch_size, channels, height, width = outputs.shape
            outputs = (
                outputs.unsqueeze(1)
                .expand(batch_size, self.num_crops, channels, height, width)
                .reshape(-1, channels, height, width)
            )
        return outputs

    def forward_out(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_crops <= 1:
            return inputs
        batch_size = inputs.shape[0] // self.num_crops
        outputs = tu.reshape_dimensions(
            inputs,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_crops),
        )
        return outputs.mean(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_in(inputs)


def crop_image_from_indices(
    images: torch.Tensor,
    crop_indices: torch.Tensor,
    crop_height: int,
    crop_width: int,
) -> torch.Tensor:
    assert crop_indices.shape[-1] == 2
    image_ndim = images.ndim
    indices_ndim = crop_indices.ndim
    assert image_ndim in {indices_ndim + 1, indices_ndim + 2}

    padded = False
    if image_ndim == indices_ndim + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        padded = True

    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_channels, image_height, image_width = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    assert bool((crop_indices[..., 0] >= 0).all())
    assert bool((crop_indices[..., 0] < (image_height - crop_height)).all())
    assert bool((crop_indices[..., 1] >= 0).all())
    assert bool((crop_indices[..., 1] < (image_width - crop_width)).all())

    crop_ind_grid_h = tu.unsqueeze_expand_at(
        torch.arange(crop_height, device=device),
        size=crop_width,
        dim=-1,
    )
    crop_ind_grid_w = tu.unsqueeze_expand_at(
        torch.arange(crop_width, device=device),
        size=crop_height,
        dim=0,
    )
    crop_grid = torch.cat(
        (crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)),
        dim=-1,
    )

    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_grid.reshape(
        grid_reshape
    )
    all_crop_inds = all_crop_inds[..., 0] * image_width + all_crop_inds[..., 1]
    all_crop_inds = tu.unsqueeze_expand_at(all_crop_inds, size=image_channels, dim=-3)
    all_crop_inds = tu.flatten(all_crop_inds, begin_axis=-2)

    images_to_crop = tu.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = tu.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    reshape_axis = crops.ndim - 1
    crops = tu.reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )
    if padded:
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(
    images: torch.Tensor,
    crop_height: int,
    crop_width: int,
    num_crops: int,
    pos_enc: bool = False,
):
    source_images = images
    if pos_enc:
        height, width = source_images.shape[-2:]
        pos_y, pos_x = torch.meshgrid(
            torch.arange(height, device=source_images.device),
            torch.arange(width, device=source_images.device),
            indexing="ij",
        )
        pos_y = pos_y.float() / float(height)
        pos_x = pos_x.float() / float(width)
        position_enc = torch.stack((pos_y, pos_x))
        leading_shape = source_images.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)
        source_images = torch.cat((source_images, position_enc), dim=-3)

    image_height, image_width = source_images.shape[-2:]
    max_sample_h = image_height - crop_height
    max_sample_w = image_width - crop_width
    crop_inds_h = (
        max_sample_h
        * torch.rand(*source_images.shape[:-3], num_crops, device=source_images.device)
    ).long()
    crop_inds_w = (
        max_sample_w
        * torch.rand(*source_images.shape[:-3], num_crops, device=source_images.device)
    ).long()
    crop_inds = torch.cat(
        (crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)),
        dim=-1,
    )
    crops = crop_image_from_indices(
        images=source_images,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )
    return crops, crop_inds
