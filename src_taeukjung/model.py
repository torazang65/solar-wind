import torch
from torch import nn
from torch.nn import functional as F
from config import (
    IMAGE_SIZE,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
    SPATIAL_FEATURE_SIZE,
)

def make_solar_disk_mask(height, width, center_fraction, radius_fraction, device=None, dtype=None):
    center_y = (height - 1) * center_fraction[0]
    center_x = (width - 1) * center_fraction[1]
    radius = min(height, width) * radius_fraction
    y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    return mask.to(dtype=dtype or torch.float32).view(1, 1, 1, height, width)

def spatial_block_average(x, output_size):
    batch, channels, time, height, width = x.shape
    if height == output_size and width == output_size:
        return x
    if height % output_size != 0 or width % output_size != 0:
        raise ValueError(
            f"feature map {(height, width)} must be divisible by {output_size}; "
            "adjust IMAGE_SIZE or SPATIAL_FEATURE_SIZE"
        )
    kernel_h = height // output_size
    kernel_w = width // output_size
    x = x.reshape(batch, channels, time, output_size, kernel_h, output_size, kernel_w)
    return x.mean(dim=(4, 6))

class Inception3D(nn.Module):
    def __init__(self, in_channels, branch_channels=32):
        super().__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1), nn.ReLU(inplace=True)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1), nn.ReLU(inplace=True),
            nn.Conv3d(branch_channels, branch_channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, 1), nn.ReLU(inplace=True),
            nn.Conv3d(branch_channels, branch_channels, (1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d((1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.Conv3d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch_1(x), self.branch_3(x), self.branch_5(x), self.branch_pool(x)],
            dim=1,
        )

class SolarWindBaseline(nn.Module):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        spatial_feature_size=SPATIAL_FEATURE_SIZE,
    ):
        super().__init__()
        self.apply_solar_disk_mask = apply_solar_disk_mask
        self.solar_disk_center_fraction = solar_disk_center_fraction
        self.solar_disk_radius_fraction = solar_disk_radius_fraction
        if apply_solar_disk_mask:
            mask = make_solar_disk_mask(
                image_size,
                image_size,
                solar_disk_center_fraction,
                solar_disk_radius_fraction,
            )
        else:
            mask = torch.ones(1, 1, 1, 1, 1)
        self.register_buffer("solar_disk_mask", mask, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv3d(2, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        blocks = []
        in_channels = 32
        for _ in range(3):
            blocks.extend([
                Inception3D(in_channels, 32),
                nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ])
            in_channels = 128
        self.image_encoder = nn.Sequential(*blocks)
        self.spatial_feature_size = spatial_feature_size
        self.image_lstm = nn.LSTM(
            input_size=128 * spatial_feature_size * spatial_feature_size,
            hidden_size=128,
            batch_first=True,
        )
        self.wind_encoder = nn.Sequential(
            nn.Linear(20, 128), nn.SELU(inplace=True),
            nn.Linear(128, 64), nn.SELU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(inplace=True), nn.Linear(64, 12)
        )

    def forward(self, images, wind):
        if self.apply_solar_disk_mask:
            if self.solar_disk_mask.shape[-2:] == images.shape[-2:]:
                mask = self.solar_disk_mask.to(device=images.device, dtype=images.dtype)
            else:
                mask = make_solar_disk_mask(
                    images.shape[-2],
                    images.shape[-1],
                    self.solar_disk_center_fraction,
                    self.solar_disk_radius_fraction,
                    device=images.device,
                    dtype=images.dtype,
                )
            images = images * mask

        image_features = images.permute(0, 2, 1, 3, 4).contiguous()
        image_features = self.stem(image_features)
        image_features = self.image_encoder(image_features)
        image_features = spatial_block_average(image_features, self.spatial_feature_size)
        image_features = image_features.permute(0, 2, 1, 3, 4).flatten(2)

        _, (hidden, _) = self.image_lstm(image_features)
        image_features = F.relu(hidden[-1])
        wind_features = self.wind_encoder(wind)
        return self.head(torch.cat([image_features, wind_features], dim=1))
