import torch
from torch import nn

from config import (
    IMAGE_SIZE,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
)
from model_baseline_v2_1 import SolarWindBaselineTransformerV21
from model_solar_probabilistic import OBSERVED_STEPS


class FactorizedSpatialTemporalBlock(nn.Module):
    """Attend over the 4x4 disk grid and time without flattening both axes."""

    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.spatial_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _attend(self, sequence, attention, norm):
        normalized = norm(sequence)
        attended, _ = attention(
            normalized, normalized, normalized, need_weights=False
        )
        return sequence + self.dropout(attended)

    def forward(self, memory):
        batch, steps, height, width, d_model = memory.shape
        cells = height * width

        spatial = memory.reshape(batch * steps, cells, d_model)
        spatial = self._attend(
            spatial, self.spatial_attention, self.spatial_norm
        )
        memory = spatial.reshape(batch, steps, height, width, d_model)

        temporal = memory.reshape(batch, steps, cells, d_model).permute(
            0, 2, 1, 3
        ).reshape(batch * cells, steps, d_model)
        temporal = self._attend(
            temporal, self.temporal_attention, self.temporal_norm
        )
        memory = temporal.reshape(batch, cells, steps, d_model).permute(
            0, 2, 1, 3
        ).reshape(batch, steps, height, width, d_model)

        normalized = self.ffn_norm(memory)
        return memory + self.dropout(self.ffn(normalized))


class SolarWindBaselineSpatialTransformerV22(SolarWindBaselineTransformerV21):
    """V2.1 with all 4x4 CNN cells retained as Transformer memory tokens."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        solar_cea_radius_fraction=None,
        latitude_bins=4,
        longitude_bins=4,
        d_model=128,
        wind_dim=64,
        nhead=4,
        encoder_layers=1,
        ff_dim=256,
        dropout=0.20,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
        baseline_residual_scale=None,
        delta_gain=4.0,
        image_time_mask_probability=0.15,
        image_modality_drop_probability=0.25,
        timing_prior_strength=0.0,
        timing_prior_sigma_hours=36.0,
        residual_cap_multiplier=1.5,
    ):
        super().__init__(
            image_size=image_size,
            apply_solar_disk_mask=apply_solar_disk_mask,
            solar_disk_center_fraction=solar_disk_center_fraction,
            solar_disk_radius_fraction=solar_disk_radius_fraction,
            solar_cea_radius_fraction=solar_cea_radius_fraction,
            latitude_bins=latitude_bins,
            longitude_bins=longitude_bins,
            d_model=d_model,
            wind_dim=wind_dim,
            nhead=nhead,
            encoder_layers=encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_images=use_images,
            baseline_slope=baseline_slope,
            baseline_intercept=baseline_intercept,
            baseline_residual_scale=baseline_residual_scale,
            delta_gain=delta_gain,
            image_time_mask_probability=image_time_mask_probability,
            image_modality_drop_probability=image_modality_drop_probability,
            timing_prior_strength=timing_prior_strength,
            timing_prior_sigma_hours=timing_prior_sigma_hours,
            residual_cap_multiplier=residual_cap_multiplier,
        )
        del self.image_projection
        del self.temporal_encoder

        self.spatial_height = int(latitude_bins)
        self.spatial_width = int(longitude_bins)
        self.memory_spatial_tokens = self.spatial_height * self.spatial_width
        self.image_cell_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )
        self.row_position = nn.Parameter(
            torch.randn(1, 1, self.spatial_height, 1, d_model) * 0.02
        )
        self.column_position = nn.Parameter(
            torch.randn(1, 1, 1, self.spatial_width, d_model) * 0.02
        )
        self.factorized_blocks = nn.ModuleList(
            [
                FactorizedSpatialTemporalBlock(
                    d_model, nhead, ff_dim, dropout
                )
                for _ in range(encoder_layers)
            ]
        )
        self.factorized_output_norm = nn.LayerNorm(d_model)

    def encode_images(self, images):
        features, masked, delta = self.image_encoder(images)
        batch = features.shape[0]
        features = features.reshape(
            batch,
            OBSERVED_STEPS,
            128,
            self.spatial_height,
            self.spatial_width,
        ).permute(0, 1, 3, 4, 2)
        memory = self.image_cell_projection(features)

        if self.training and self.image_time_mask_probability > 0.0:
            time_keep = (
                torch.rand(
                    batch,
                    OBSERVED_STEPS,
                    1,
                    1,
                    1,
                    device=memory.device,
                )
                >= self.image_time_mask_probability
            ).to(dtype=memory.dtype)
            memory = memory * time_keep
        keep = self._image_keep_mask(batch, memory.device, memory.dtype)
        memory = memory * keep.reshape(batch, 1, 1, 1, 1)

        time_position = self.observed_time_encoding.reshape(
            1, OBSERVED_STEPS, 1, 1, self.d_model
        )
        memory = (
            memory
            + self.position_scale * time_position
            + self.row_position
            + self.column_position
        )
        for block in self.factorized_blocks:
            memory = block(memory)
        memory = self.factorized_output_norm(memory)
        memory = memory.reshape(
            batch, OBSERVED_STEPS * self.memory_spatial_tokens, self.d_model
        )
        return memory, keep, masked, delta

    def transformer_value_count(self):
        return OBSERVED_STEPS * self.memory_spatial_tokens * self.d_model

    def encoder_attention_score_count(self):
        spatial = OBSERVED_STEPS * self.memory_spatial_tokens**2
        temporal = self.memory_spatial_tokens * OBSERVED_STEPS**2
        return len(self.factorized_blocks) * (spatial + temporal)
