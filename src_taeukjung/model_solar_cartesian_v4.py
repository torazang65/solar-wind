import torch
from torch import nn
from torch.nn import functional as F

from config import (
    IMAGE_SIZE,
    SOLAR_CEA_RADIUS_FRACTION,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
)
from model_solar_geometry_v3 import SolarWindGeometryTransformerV3
from model_solar_probabilistic import (
    DualPolarityDownsample,
    MultiScaleSolarBlock,
    channel_norm,
)


class SolarCartesianEncoderV4(nn.Module):
    """Keep the observed disk projection and expose centered disk coordinates."""

    def __init__(
        self,
        image_size,
        radius_fraction,
        output_channels,
        spatial_height=4,
        spatial_width=8,
        visual_dropout=0.1,
    ):
        super().__init__()
        if not 0.0 < radius_fraction <= 0.5:
            raise ValueError("expected 0 < radius_fraction <= 0.5")

        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        center = (image_size - 1) * 0.5
        radius = image_size * radius_fraction
        y = (torch.arange(image_size, dtype=torch.float32) - center) / radius
        x = (torch.arange(image_size, dtype=torch.float32) - center) / radius
        y, x = torch.meshgrid(y, x, indexing="ij")
        radius_squared = x.square() + y.square()
        mask = (radius_squared <= 1.0).float()
        mu = torch.sqrt((1.0 - radius_squared).clamp_min(0.0)) * mask
        coordinates = torch.stack(
            [x.clamp(-1.0, 1.0) * mask, y.abs().clamp(0.0, 1.0) * mask, mu],
            dim=0,
        )
        self.register_buffer("disk_mask", mask.view(1, 1, image_size, image_size))
        self.register_buffer("coordinates", coordinates.unsqueeze(0))

        self.stem = nn.Sequential(
            nn.Conv2d(7, 24, kernel_size=5, stride=2, padding=2),
            channel_norm(24),
            nn.GELU(),
        )
        self.block_1 = MultiScaleSolarBlock(24, 48)
        self.downsample_1 = DualPolarityDownsample(48)
        self.block_2 = MultiScaleSolarBlock(48, output_channels)
        self.downsample_2 = DualPolarityDownsample(output_channels)
        self.feature_dropout = nn.Dropout2d(visual_dropout)

    def forward(self, images):
        batch, steps, channels, height, width = images.shape
        mask = self.disk_mask.to(device=images.device, dtype=images.dtype)
        projected = images.reshape(batch * steps, channels, height, width) * mask
        coordinates = self.coordinates.to(device=images.device, dtype=images.dtype)
        coordinates = coordinates.expand(batch * steps, -1, -1, -1)

        mu = coordinates[:, 2:3].clamp(0.0, 1.0)
        reference = (projected * mu).sum(dim=(-2, -1), keepdim=True)
        reference = reference / mu.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        relative_darkness = F.relu(reference - projected)
        relative_darkness = relative_darkness / reference.clamp_min(1.0 / 255.0)
        relative_darkness = (relative_darkness * torch.sqrt(mu)).clamp(0.0, 1.0)

        features = torch.cat([projected, relative_darkness, coordinates], dim=1)
        features = self.stem(features)
        features = self.feature_dropout(self.downsample_1(self.block_1(features)))
        features = self.feature_dropout(self.downsample_2(self.block_2(features)))
        features = F.adaptive_avg_pool2d(
            features, (self.spatial_height, self.spatial_width)
        )
        features = features.flatten(2).transpose(1, 2)
        return features.reshape(
            batch,
            steps,
            self.spatial_height * self.spatial_width,
            -1,
        )


class SolarWindCartesianTransformerV4(SolarWindGeometryTransformerV3):
    """V3 temporal model with the original Cartesian disk projection."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        solar_cea_radius_fraction=SOLAR_CEA_RADIUS_FRACTION,
        spatial_height=4,
        spatial_width=8,
        d_model=96,
        wind_dim=24,
        nhead=8,
        encoder_layers=1,
        ff_dim=192,
        dropout=0.25,
        visual_dropout=0.1,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
    ):
        super().__init__(
            image_size=image_size,
            apply_solar_disk_mask=apply_solar_disk_mask,
            solar_disk_center_fraction=solar_disk_center_fraction,
            solar_disk_radius_fraction=solar_disk_radius_fraction,
            solar_cea_radius_fraction=solar_cea_radius_fraction,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            d_model=d_model,
            wind_dim=wind_dim,
            nhead=nhead,
            encoder_layers=encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            visual_dropout=visual_dropout,
            use_images=use_images,
            baseline_slope=baseline_slope,
            baseline_intercept=baseline_intercept,
        )
        self.image_encoder = SolarCartesianEncoderV4(
            image_size,
            solar_disk_radius_fraction,
            self.image_feature_dim,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            visual_dropout=visual_dropout,
        )
