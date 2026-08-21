import torch
from torch import nn
from torch.nn import functional as F

from model_solar_source_map_v11 import (
    MPSCompatibleSpatialPool,
    make_soft_solar_disk_mask,
)
from src_torazang65.model import Inception3D, conv_block


FORECAST_STEPS = 12
OBSERVED_STEPS = 20
ARCHITECTURE_NAME = "SolarWindSourceMapV11_1"
FILE_STEM = "solar_source_map_v11_1"


class SolarWindSourceMapV11_1(nn.Module):
    """Seokho V7 propagation-only source map with solar-disk masking."""

    def __init__(
        self,
        image_size=64,
        use_images=True,
        d_model=128,
        dropout=0.10,
        time_mask_prob=0.15,
        modality_drop_prob=0.25,
        delta_gain=1.0,
        apply_solar_disk_mask=True,
        solar_disk_center_fraction=(0.5, 0.5),
        solar_disk_radius_fraction=0.49,
        solar_disk_edge_pixels=1.5,
        kernel_sigma_hours=12.0,
        transit_residual_hours=24.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 <= time_mask_prob < 1.0:
            raise ValueError("time_mask_prob must be in [0, 1)")
        if not 0.0 <= modality_drop_prob < 1.0:
            raise ValueError("modality_drop_prob must be in [0, 1)")
        if delta_gain <= 0.0 or kernel_sigma_hours <= 0.0:
            raise ValueError("delta_gain and kernel_sigma_hours must be positive")
        if transit_residual_hours < 0.0:
            raise ValueError("transit_residual_hours must be nonnegative")

        self.image_size = int(image_size)
        self.use_images = bool(use_images)
        self.d_model = int(d_model)
        self.time_mask_prob = float(time_mask_prob)
        self.modality_drop_prob = float(modality_drop_prob)
        self.delta_gain = float(delta_gain)
        self.apply_solar_disk_mask = bool(apply_solar_disk_mask)
        self.kernel_sigma_hours = float(kernel_sigma_hours)
        self.transit_residual_hours = float(transit_residual_hours)

        self.register_buffer(
            "solar_disk_mask",
            make_soft_solar_disk_mask(
                image_size,
                center_fraction=solar_disk_center_fraction,
                radius_fraction=solar_disk_radius_fraction,
                edge_pixels=solar_disk_edge_pixels,
            ),
        )
        self.register_buffer(
            "image_age_hours",
            torch.arange(OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "hindcast_hours", torch.arange(-12, 1, dtype=torch.float32) * 6.0
        )
        self.register_buffer(
            "horizon_hours",
            torch.arange(1, FORECAST_STEPS + 1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "cell_lon_deg", torch.tensor([-67.5, -22.5, 22.5, 67.5])
        )
        self.register_buffer("cell_lat_norm", torch.tensor([0.5, -0.5]))
        self.omega_deg_per_hour = 360.0 / (27.2753 * 24.0)

        self.stem = nn.Sequential(
            conv_block(4, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        blocks = []
        in_channels = 32
        for _ in range(3):
            blocks.extend(
                [
                    Inception3D(in_channels, 32),
                    nn.MaxPool3d(
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                    ),
                ]
            )
            in_channels = 128
        self.image_encoder = nn.Sequential(*blocks)
        self.spatial_pool = MPSCompatibleSpatialPool()
        self.image_projection = nn.Linear(128 * 2 * 4, d_model)

        head_input_dim = 128 + 2
        self.source_speed_head = nn.Linear(head_input_dim, 1)
        self.source_gate_head = nn.Linear(head_input_dim, 1)
        self.transit_residual_head = nn.Linear(head_input_dim, 1)
        self.lon_offset_head = nn.Linear(head_input_dim, 1)

        self.dist_eff_raw = nn.Parameter(torch.tensor(-0.144))
        self.fallback_weight_raw = nn.Parameter(torch.tensor(0.5413))
        self.climatology = nn.Parameter(torch.tensor(0.43))
        self.reversion_logit = nn.Parameter(torch.full((FORECAST_STEPS,), -4.0))

        self.fusion_image_proj = nn.Linear(d_model, 16)
        self.surge_head = nn.Sequential(
            nn.Linear(2 * d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.fusion_gate_head = nn.Linear(16 + 4 + 1, FORECAST_STEPS)

        nn.init.zeros_(self.source_speed_head.weight)
        nn.init.constant_(self.source_speed_head.bias, -0.96)
        nn.init.zeros_(self.source_gate_head.weight)
        nn.init.zeros_(self.source_gate_head.bias)
        nn.init.zeros_(self.transit_residual_head.weight)
        nn.init.zeros_(self.transit_residual_head.bias)
        nn.init.zeros_(self.lon_offset_head.weight)
        nn.init.zeros_(self.lon_offset_head.bias)
        nn.init.zeros_(self.fusion_gate_head.weight)
        nn.init.constant_(self.fusion_gate_head.bias, -2.0)

        self._last_diagnostics = {}

    def effective_distance(self):
        return 30.0 + 25.0 * torch.sigmoid(self.dist_eff_raw)

    def _encode_images(self, images):
        if self.apply_solar_disk_mask:
            images = images * self.solar_disk_mask.to(dtype=images.dtype)
        difference = torch.zeros_like(images)
        difference[:, 1:] = images[:, 1:] - images[:, :-1]
        image_channels = torch.cat(
            [images, self.delta_gain * difference], dim=2
        )
        features = image_channels.permute(0, 2, 1, 3, 4).contiguous()
        features = self.image_encoder(self.stem(features))
        features = self.spatial_pool(features)
        cell_features = features.permute(0, 2, 3, 4, 1).contiguous()
        image_tokens = self.image_projection(
            features.permute(0, 2, 1, 3, 4).flatten(2)
        )
        return cell_features, image_tokens

    def _apply_image_augmentation(self, cell_features, image_tokens):
        batch_size = image_tokens.shape[0]
        time_keep = torch.ones(
            batch_size,
            OBSERVED_STEPS,
            1,
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )
        if self.training and self.time_mask_prob > 0.0:
            time_keep = (
                torch.rand(
                    batch_size,
                    OBSERVED_STEPS,
                    1,
                    device=image_tokens.device,
                )
                >= self.time_mask_prob
            ).to(dtype=image_tokens.dtype)
            image_tokens = image_tokens * time_keep
            cell_features = cell_features * time_keep.view(
                batch_size, OBSERVED_STEPS, 1, 1, 1
            )

        image_keep = torch.ones(
            batch_size, device=image_tokens.device, dtype=image_tokens.dtype
        )
        if self.training and self.modality_drop_prob > 0.0:
            image_keep = (
                torch.rand(batch_size, device=image_tokens.device)
                >= self.modality_drop_prob
            ).to(dtype=image_tokens.dtype)
            image_tokens = image_tokens * image_keep.view(batch_size, 1, 1)
            cell_features = cell_features * image_keep.view(
                batch_size, 1, 1, 1, 1
            )
        return cell_features, image_tokens, image_keep

    def _source_map(self, cell_features, image_tokens, image_keep, wind):
        batch_size = wind.shape[0]
        latitude = self.cell_lat_norm.view(1, 1, 2, 1, 1).expand(
            batch_size, OBSERVED_STEPS, 2, 4, 1
        )
        longitude = (self.cell_lon_deg / 90.0).view(1, 1, 1, 4, 1).expand(
            batch_size, OBSERVED_STEPS, 2, 4, 1
        )
        head_input = torch.cat([cell_features, latitude, longitude], dim=-1)
        source_speed = 0.25 + 0.65 * torch.sigmoid(
            self.source_speed_head(head_input)
        ).squeeze(-1)
        source_gate = F.softplus(self.source_gate_head(head_input)).squeeze(-1)
        transit_residual = torch.tanh(
            self.transit_residual_head(head_input)
        ).squeeze(-1)
        source_longitude = self.cell_lon_deg.view(1, 1, 1, 4) + 22.5 * torch.tanh(
            self.lon_offset_head(head_input)
        ).squeeze(-1)

        rotation_wait = -source_longitude / self.omega_deg_per_hour
        transit = (
            self.effective_distance().to(dtype=source_speed.dtype) / source_speed
            + self.transit_residual_hours * transit_residual
        )
        arrival = (
            rotation_wait
            + transit
            - self.image_age_hours.view(1, OBSERVED_STEPS, 1, 1)
        )
        time_grid = torch.cat([self.hindcast_hours, self.horizon_hours])
        kernel = torch.exp(
            -(
                time_grid.view(1, 1, 1, 1, -1) - arrival.unsqueeze(-1)
            ).square()
            / (2.0 * self.kernel_sigma_hours**2)
        )
        source_weight = source_gate.unsqueeze(-1) * kernel
        weight_sum = source_weight.sum(dim=(1, 2, 3))
        fallback = F.softplus(self.fallback_weight_raw)
        source_prediction = (
            (source_weight * source_speed.unsqueeze(-1)).sum(dim=(1, 2, 3))
            + fallback * self.climatology
        ) / (weight_sum + fallback)
        coverage = weight_sum / (weight_sum + fallback)

        image_summary = F.gelu(
            self.fusion_image_proj(image_tokens.mean(dim=1))
        )
        surge_logit = self.surge_head(
            torch.cat(
                [
                    image_tokens.mean(dim=1),
                    image_tokens[:, -5:].mean(dim=1)
                    - image_tokens[:, :5].mean(dim=1),
                ],
                dim=1,
            )
        )
        surge_probability = torch.sigmoid(surge_logit)
        wind_summary = torch.stack(
            [
                wind[:, -1],
                wind.mean(dim=1),
                wind.std(dim=1),
                wind[:, -1] - wind[:, 0],
            ],
            dim=1,
        )
        fusion_alpha = torch.sigmoid(
            self.fusion_gate_head(
                torch.cat(
                    [image_summary, wind_summary, surge_probability], dim=1
                )
            )
        )
        return {
            "hindcast": source_prediction[:, :13],
            "future": source_prediction[:, 13:],
            "fusion_alpha": fusion_alpha,
            "image_keep": image_keep,
            "transit_residual": transit_residual,
            "source_weight": source_weight,
            "source_speed": source_speed,
            "source_gate": source_gate,
            "source_longitude": source_longitude,
            "arrival": arrival,
            "coverage": coverage,
            "surge_logit": surge_logit,
            "surge_probability": surge_probability,
        }

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_aux=False,
    ):
        batch_size = wind.shape[0]
        beta = torch.sigmoid(self.reversion_logit).to(dtype=wind.dtype)
        base = wind[:, -1:] + beta * (self.climatology - wind[:, -1:])

        if self.use_images:
            cell_features, image_tokens = self._encode_images(images)
            cell_features, image_tokens, image_keep = (
                self._apply_image_augmentation(cell_features, image_tokens)
            )
            source = self._source_map(
                cell_features, image_tokens, image_keep, wind
            )
            propagation = source["fusion_alpha"] * (source["future"] - base)
            prediction = base + propagation
            self._last_diagnostics = {
                "source_speed_mean_kms": source["source_speed"].mean() * 1000.0,
                "source_speed_std_kms": source["source_speed"].std() * 1000.0,
                "arrival_mean_h": source["arrival"].mean(),
                "arrival_std_h": source["arrival"].std(),
                "source_gate_mean": source["source_gate"].mean(),
                "coverage_hind_mean": source["coverage"][:, :13].mean(),
                "coverage_future_mean": source["coverage"][:, 13:].mean(),
                "fusion_alpha_mean": source["fusion_alpha"].mean(),
                "surge_probability_mean": source["surge_probability"].mean(),
                "beta_mean": beta.mean(),
                "dist_eff_h": self.effective_distance(),
            }
        else:
            source = None
            propagation = torch.zeros_like(base)
            prediction = base
            self._last_diagnostics = {
                "beta_mean": beta.mean(),
                "dist_eff_h": self.effective_distance(),
            }

        components = {
            "base": base,
            "source_prediction": (
                source["future"]
                if source is not None
                else self.climatology.expand(batch_size, FORECAST_STEPS)
            ),
            "fusion_alpha": (
                source["fusion_alpha"]
                if source is not None
                else torch.zeros_like(base)
            ),
            "propagation_residual": propagation,
            "surge_probability": (
                source["surge_probability"]
                if source is not None
                else torch.zeros(batch_size, 1, device=wind.device, dtype=wind.dtype)
            ),
        }
        aux = {
            "hindcast": source["hindcast"] if source is not None else None,
            "image_keep": source["image_keep"] if source is not None else None,
            "transit_residual": (
                source["transit_residual"] if source is not None else None
            ),
            "source_weight": source["source_weight"] if source is not None else None,
            "surge_logit": source["surge_logit"] if source is not None else None,
        }
        if return_components and return_aux:
            return prediction, components, aux
        if return_components:
            return prediction, components
        if return_aux:
            return prediction, aux
        return prediction

    def training_diagnostics(self):
        return self._last_diagnostics
