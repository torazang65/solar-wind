import torch
from torch import nn

from model_solar_lstm_v12 import (
    FORECAST_STEPS,
    OBSERVED_STEPS,
    SolarWindLagLSTMV12,
)


ARCHITECTURE_NAME = "SolarWindNativeProfileLSTMV16"
FILE_STEM = "solar_native_profile_lstm_v16"


class SolarWindNativeProfileLSTMV16(SolarWindLagLSTMV12):
    """Native-width longitude profiles with the guarded V12 LSTM back end."""

    def __init__(
        self,
        column_dim=16,
        longitude_kernel_size=5,
        scramble_images=False,
        disable_wind_residual=True,
        **kwargs,
    ):
        if column_dim <= 0:
            raise ValueError("column_dim must be positive")
        if longitude_kernel_size <= 0 or longitude_kernel_size % 2 == 0:
            raise ValueError("longitude_kernel_size must be positive and odd")
        image_size = int(kwargs.get("image_size", 64))
        kwargs["grid_rows"] = 1
        kwargs["grid_columns"] = image_size
        kwargs["cell_dim"] = int(column_dim)
        super().__init__(**kwargs)

        self.column_dim = int(column_dim)
        self.longitude_kernel_size = int(longitude_kernel_size)
        self.scramble_images = bool(scramble_images)
        self.disable_wind_residual = bool(disable_wind_residual)

        # Remove every spatial resizing module allocated by the V12 parent.
        self.stem = None
        self.image_blocks = nn.ModuleList()
        self.spatial_pool = None
        self.cell_projection = None
        self.coordinate_projection = None
        self.cell_norm = None
        self.spatial_importance_head = None

        # 2 channels x 4 native-column statistics, plus signed differences.
        profile_dim = 2 * 4 * 2
        self.column_projection = nn.Sequential(
            nn.Linear(profile_dim, self.column_dim),
            nn.GELU(),
            nn.LayerNorm(self.column_dim),
        )
        padding = self.longitude_kernel_size // 2
        self.longitude_mixer = nn.Sequential(
            nn.Conv1d(
                self.column_dim,
                self.column_dim,
                kernel_size=self.longitude_kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.column_dim,
                self.column_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )
        self.longitude_embedding = nn.Parameter(
            torch.empty(self.image_size, self.column_dim)
        )
        self.column_importance_head = nn.Linear(self.column_dim, 1)
        self.frame_projection = nn.Sequential(
            nn.Linear(self.image_size * self.column_dim, self.frame_dim),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.15)),
        )
        nn.init.normal_(self.longitude_embedding, std=0.02)
        nn.init.zeros_(self.column_importance_head.weight)
        nn.init.zeros_(self.column_importance_head.bias)

        if self.disable_wind_residual:
            nn.init.zeros_(self.wind_residual_head.weight)
            nn.init.zeros_(self.wind_residual_head.bias)
            for parameter in self.wind_residual_head.parameters():
                parameter.requires_grad_(False)

    def _native_column_statistics(self, images):
        if self.scramble_images:
            images = torch.flip(images, dims=(1, 4))
        mask = self.solar_disk_mask.to(dtype=images.dtype)
        weight = mask.view(1, 1, 1, self.image_size, self.image_size)
        coverage = weight.sum(dim=-2).clamp_min(1e-6)
        mean = (images * weight).sum(dim=-2) / coverage
        variance = (
            (images - mean.unsqueeze(-2)).square() * weight
        ).sum(dim=-2) / coverage
        valid = (mask > 0.05).view(
            1, 1, 1, self.image_size, self.image_size
        )
        minimum = images.masked_fill(~valid, float("inf")).amin(dim=-2)
        maximum = images.masked_fill(~valid, float("-inf")).amax(dim=-2)
        valid_column = valid.any(dim=-2)
        minimum = torch.where(valid_column, minimum, torch.zeros_like(minimum))
        maximum = torch.where(valid_column, maximum, torch.zeros_like(maximum))
        statistics = torch.stack(
            [mean, minimum, maximum, variance.clamp_min(0.0).sqrt()], dim=-1
        )
        return statistics.permute(0, 1, 3, 2, 4).flatten(-2)

    def _encode_images(self, images):
        profile = self._native_column_statistics(images)
        differences = torch.zeros_like(profile)
        differences[:, 1:] = profile[:, 1:] - profile[:, :-1]
        profile = torch.cat([profile, self.delta_gain * differences], dim=-1)
        columns = self.column_projection(profile)
        columns = columns + self.longitude_embedding.to(dtype=columns.dtype)
        batch_size = columns.shape[0]
        mixed = self.longitude_mixer(
            columns.reshape(-1, self.image_size, self.column_dim).transpose(1, 2)
        ).transpose(1, 2)
        columns = columns + mixed.view(
            batch_size, OBSERVED_STEPS, self.image_size, self.column_dim
        )
        column_logits = self.column_importance_head(columns).squeeze(-1)
        column_attention = torch.softmax(column_logits, dim=-1)
        frame_tokens = self.frame_norm(
            self.frame_projection(columns.flatten(2))
        )
        return frame_tokens, column_attention.unsqueeze(2)
