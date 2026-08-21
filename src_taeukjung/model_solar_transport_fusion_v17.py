import math

import torch
from torch import nn

from model_solar_source_map_v11 import make_soft_solar_disk_mask


OBSERVED_STEPS = 20
HINDCAST_STEPS = 10
FORECAST_STEPS = 12
ARCHITECTURE_NAME = "SolarWindTransportFusionV17"
FILE_STEM = "solar_transport_fusion_v17"


class SolarWindTransportFusionV17(nn.Module):
    """Native longitude transport bank followed by a guarded AR fusion head."""

    def __init__(
        self,
        image_size=64,
        use_images=True,
        ar_coefficients=None,
        ar_intercept=0.0,
        baseline_residual_scale=None,
        column_dim=32,
        longitude_kernel_size=5,
        speed_experts_kms=(300.0, 400.0, 500.0, 650.0, 800.0),
        transport_sigma_hours=15.0,
        effective_distance_hours_at_1000_kms=41.6,
        solar_rotation_days=27.27,
        minimum_delay_hours=48.0,
        maximum_delay_hours=144.0,
        transport_strength=0.50,
        correction_cap_multiplier=1.0,
        dropout=0.10,
        time_mask_prob=0.10,
        modality_drop_prob=0.10,
        delta_gain=1.0,
        scramble_images=False,
        apply_solar_disk_mask=True,
        solar_disk_center_fraction=(0.5, 0.5),
        solar_disk_radius_fraction=0.49,
        solar_disk_edge_pixels=1.5,
    ):
        super().__init__()
        if ar_coefficients is None:
            ar_coefficients = [0.0, 1.0]
        if baseline_residual_scale is None:
            baseline_residual_scale = [0.08] * FORECAST_STEPS
        coefficients = torch.as_tensor(ar_coefficients, dtype=torch.float32)
        residual_scale = torch.as_tensor(
            baseline_residual_scale, dtype=torch.float32
        )
        expert_speed = torch.as_tensor(speed_experts_kms, dtype=torch.float32)
        if coefficients.ndim != 1 or not 1 <= len(coefficients) <= OBSERVED_STEPS:
            raise ValueError("invalid AR coefficients")
        if residual_scale.shape != (FORECAST_STEPS,) or torch.any(
            residual_scale <= 0
        ):
            raise ValueError("baseline_residual_scale must contain 12 positives")
        if expert_speed.ndim != 1 or len(expert_speed) < 2 or torch.any(
            expert_speed <= 0
        ):
            raise ValueError("speed_experts_kms must contain at least two positives")
        if column_dim <= 0:
            raise ValueError("column_dim must be positive")
        if longitude_kernel_size <= 0 or longitude_kernel_size % 2 == 0:
            raise ValueError("longitude_kernel_size must be positive and odd")
        if transport_sigma_hours <= 0.0:
            raise ValueError("transport_sigma_hours must be positive")
        if not 0.0 < minimum_delay_hours < maximum_delay_hours:
            raise ValueError("invalid physical delay bounds")
        if correction_cap_multiplier <= 0.0:
            raise ValueError("correction_cap_multiplier must be positive")
        if not 0.0 <= time_mask_prob < 1.0:
            raise ValueError("time_mask_prob must be in [0, 1)")
        if not 0.0 <= modality_drop_prob < 1.0:
            raise ValueError("modality_drop_prob must be in [0, 1)")

        self.image_size = int(image_size)
        self.use_images = bool(use_images)
        self.column_dim = int(column_dim)
        self.longitude_kernel_size = int(longitude_kernel_size)
        self.transport_sigma_hours = float(transport_sigma_hours)
        self.effective_distance_hours_at_1000_kms = float(
            effective_distance_hours_at_1000_kms
        )
        self.solar_rotation_days = float(solar_rotation_days)
        self.minimum_delay_hours = float(minimum_delay_hours)
        self.maximum_delay_hours = float(maximum_delay_hours)
        self.transport_strength = float(transport_strength)
        self.correction_cap_multiplier = float(correction_cap_multiplier)
        self.time_mask_prob = float(time_mask_prob)
        self.modality_drop_prob = float(modality_drop_prob)
        self.delta_gain = float(delta_gain)
        self.scramble_images = bool(scramble_images)
        self.apply_solar_disk_mask = bool(apply_solar_disk_mask)

        # Compatibility attributes used by the shared inference diagnostics.
        self.grid_rows = 3
        self.grid_columns = self.image_size
        self.frame_dim = self.column_dim
        self.lstm_hidden_dim = 0
        self.lag_prior_max_strength = 0.0

        self.register_buffer("ar_coefficients", coefficients)
        self.register_buffer("ar_intercept", torch.tensor(float(ar_intercept)))
        self.register_buffer("baseline_residual_scale", residual_scale)
        self.register_buffer("speed_experts", expert_speed / 1000.0)
        self.register_buffer(
            "lag_hours",
            torch.tensor([minimum_delay_hours, maximum_delay_hours]),
        )
        self.register_buffer(
            "source_hours",
            torch.arange(-(OBSERVED_STEPS - 1), 1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "hindcast_hours",
            torch.arange(-(HINDCAST_STEPS - 1), 1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "forecast_hours",
            torch.arange(1, FORECAST_STEPS + 1, dtype=torch.float32) * 6.0,
        )
        pixel_x = (
            torch.arange(self.image_size, dtype=torch.float32)
            + 0.5
            - self.image_size / 2.0
        ) / (self.image_size * float(solar_disk_radius_fraction))
        self.register_buffer(
            "longitude_degrees",
            torch.asin(pixel_x.clamp(-1.0, 1.0)) * (180.0 / math.pi),
        )
        disk_mask = make_soft_solar_disk_mask(
            self.image_size,
            center_fraction=solar_disk_center_fraction,
            radius_fraction=solar_disk_radius_fraction,
            edge_pixels=solar_disk_edge_pixels,
        )
        if not self.apply_solar_disk_mask:
            disk_mask = torch.ones_like(disk_mask)
        latitude = torch.linspace(-1.0, 1.0, self.image_size).view(-1, 1)
        bands = torch.stack(
            [
                (latitude < -1.0 / 3.0).float().expand(-1, self.image_size),
                ((latitude >= -1.0 / 3.0) & (latitude <= 1.0 / 3.0))
                .float()
                .expand(-1, self.image_size),
                (latitude > 1.0 / 3.0).float().expand(-1, self.image_size),
            ]
        )
        self.register_buffer("latitude_band_masks", bands * disk_mask.unsqueeze(0))

        # Two channels, three latitude bands, four statistics, plus signed deltas.
        profile_dim = 2 * 3 * 4 * 2
        self.profile_projection = nn.Sequential(
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
            nn.Conv1d(self.column_dim, self.column_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.longitude_embedding = nn.Parameter(
            torch.empty(self.image_size, self.column_dim)
        )
        self.expert_head = nn.Linear(self.column_dim, len(self.speed_experts))
        self.evidence_head = nn.Linear(self.column_dim, 1)

        # Shared across horizons. Physics enters explicitly through transport - AR.
        self.fusion_head = nn.Sequential(
            nn.Linear(8, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

        nn.init.normal_(self.longitude_embedding, std=0.02)
        nn.init.normal_(self.expert_head.weight, std=0.02)
        prior = -0.5 * ((expert_speed - 430.0) / 130.0).square()
        with torch.no_grad():
            self.expert_head.bias.copy_(prior)
        nn.init.normal_(self.evidence_head.weight, std=0.02)
        nn.init.zeros_(self.evidence_head.bias)
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)
        self._last_diagnostics = {}

    @property
    def omega_deg_per_hour(self):
        return 360.0 / (self.solar_rotation_days * 24.0)

    def physical_delay_hours(self):
        transit = self.effective_distance_hours_at_1000_kms / self.speed_experts
        rotation_wait = -self.longitude_degrees / self.omega_deg_per_hour
        return (rotation_wait[:, None] + transit[None, :]).clamp(
            self.minimum_delay_hours, self.maximum_delay_hours
        )

    def _recursive_ar(self, wind):
        order = len(self.ar_coefficients)
        history = [wind[:, index] for index in range(OBSERVED_STEPS)]
        predictions = []
        for _ in range(FORECAST_STEPS):
            context = torch.stack(history[-order:], dim=1)
            next_value = self.ar_intercept.to(dtype=wind.dtype) + (
                context * self.ar_coefficients.to(dtype=wind.dtype)
            ).sum(dim=1)
            history.append(next_value)
            predictions.append(next_value)
        return torch.stack(predictions, dim=1)

    def _sample_augmentation(self, batch_size, device, dtype):
        time_keep = torch.ones(
            batch_size, OBSERVED_STEPS, device=device, dtype=dtype
        )
        if self.training and self.time_mask_prob > 0.0:
            time_keep = (
                torch.rand(batch_size, OBSERVED_STEPS, device=device)
                >= self.time_mask_prob
            ).to(dtype=dtype)
            time_keep[:, -1] = 1.0
        image_keep = torch.ones(batch_size, device=device, dtype=dtype)
        if self.training and self.modality_drop_prob > 0.0:
            image_keep = (
                torch.rand(batch_size, device=device)
                >= self.modality_drop_prob
            ).to(dtype=dtype)
        return time_keep, image_keep

    def _native_profiles(self, images):
        if self.scramble_images:
            images = torch.flip(images, dims=(1, 4))
        masks = self.latitude_band_masks.to(dtype=images.dtype)
        weight = masks.view(1, 1, 1, 3, self.image_size, self.image_size)
        expanded = images.unsqueeze(3)
        coverage = weight.sum(dim=-2).clamp_min(1e-5)
        mean = (expanded * weight).sum(dim=-2) / coverage
        variance = ((expanded - mean.unsqueeze(-2)).square() * weight).sum(
            dim=-2
        ) / coverage
        dark_fraction = (torch.sigmoid((0.35 - expanded) * 16.0) * weight).sum(
            dim=-2
        ) / coverage
        bright_fraction = (
            torch.sigmoid((expanded - 0.65) * 16.0) * weight
        ).sum(dim=-2) / coverage
        statistics = torch.stack(
            [mean, variance.clamp_min(0.0).sqrt(), dark_fraction, bright_fraction],
            dim=-1,
        )
        # B,T,C,band,longitude,statistic -> B,T,longitude,24
        return statistics.permute(0, 1, 4, 2, 3, 5).flatten(-3)

    def _encode_sources(self, images):
        profile = self._native_profiles(images)
        differences = torch.zeros_like(profile)
        differences[:, 1:] = profile[:, 1:] - profile[:, :-1]
        token = self.profile_projection(
            torch.cat([profile, self.delta_gain * differences], dim=-1)
        )
        token = token + self.longitude_embedding.to(dtype=token.dtype)
        batch_size = token.shape[0]
        mixed = self.longitude_mixer(
            token.reshape(-1, self.image_size, self.column_dim).transpose(1, 2)
        ).transpose(1, 2)
        token = token + mixed.reshape(
            batch_size, OBSERVED_STEPS, self.image_size, self.column_dim
        )
        expert_probability = torch.softmax(self.expert_head(token), dim=-1)
        evidence = 0.05 + 0.95 * torch.sigmoid(self.evidence_head(token).squeeze(-1))
        return token, expert_probability, evidence

    def _transport(self, expert_probability, evidence, time_keep, query_hours):
        dtype = expert_probability.dtype
        source_arrival = (
            self.source_hours.to(dtype=dtype).view(OBSERVED_STEPS, 1, 1)
            + self.physical_delay_hours().to(dtype=dtype).unsqueeze(0)
        )
        delta = query_hours.to(dtype=dtype).view(1, 1, 1, -1) - source_arrival.unsqueeze(-1)
        kernel = torch.exp(
            -0.5 * (delta / self.transport_sigma_hours).square()
        )
        causal_source = self.source_hours.view(OBSERVED_STEPS, 1) < query_hours.view(
            1, -1
        )
        kernel = kernel * causal_source.to(dtype=dtype).view(
            OBSERVED_STEPS, 1, 1, -1
        )
        activation = (
            evidence.unsqueeze(-1)
            * expert_probability
            * time_keep[:, :, None, None]
        )
        weight = activation.unsqueeze(-1) * kernel.unsqueeze(0)
        denominator = weight.sum(dim=(1, 2, 3)).clamp_min(1e-7)
        numerator = (
            weight * self.speed_experts.to(dtype=dtype).view(1, 1, 1, -1, 1)
        ).sum(dim=(1, 2, 3))
        transport = numerator / denominator

        normalized = weight / denominator[:, None, None, None, :]
        expected_delay = (
            normalized
            * self.physical_delay_hours()
            .to(dtype=dtype)
            .view(1, 1, self.image_size, -1, 1)
        ).sum(dim=(1, 2, 3))
        entropy = -(
            normalized.clamp_min(1e-9).log() * normalized
        ).sum(dim=(1, 2, 3)) / math.log(
            OBSERVED_STEPS * self.image_size * len(self.speed_experts)
        )
        return transport, denominator, expected_delay, entropy

    def set_stage(self, stage):
        if stage not in {"transport", "fusion", "joint"}:
            raise ValueError(f"unknown V17 training stage: {stage}")
        transport_modules = (
            self.profile_projection,
            self.longitude_mixer,
            self.expert_head,
            self.evidence_head,
        )
        transport_trainable = stage in {"transport", "joint"}
        fusion_trainable = stage in {"fusion", "joint"}
        for module in transport_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(transport_trainable)
        self.longitude_embedding.requires_grad_(transport_trainable)
        for parameter in self.fusion_head.parameters():
            parameter.requires_grad_(fusion_trainable)

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_aux=False,
        time_keep=None,
        image_keep=None,
    ):
        batch_size = wind.shape[0]
        ar_base = self._recursive_ar(wind)
        if time_keep is None or image_keep is None:
            sampled_time_keep, sampled_image_keep = self._sample_augmentation(
                batch_size, wind.device, wind.dtype
            )
            if time_keep is None:
                time_keep = sampled_time_keep
            if image_keep is None:
                image_keep = sampled_image_keep
        time_keep = torch.as_tensor(time_keep, device=wind.device, dtype=wind.dtype)
        image_keep = torch.as_tensor(
            image_keep, device=wind.device, dtype=wind.dtype
        ).flatten()
        if time_keep.shape == (batch_size, OBSERVED_STEPS, 1):
            time_keep = time_keep.squeeze(-1)
        if time_keep.shape != (batch_size, OBSERVED_STEPS):
            raise ValueError("time_keep must have shape (batch, 20)")
        if image_keep.shape != (batch_size,):
            raise ValueError("image_keep must have shape (batch,)")

        if self.use_images:
            _, expert_probability, evidence = self._encode_sources(images)
        else:
            expert_probability = torch.full(
                (
                    batch_size,
                    OBSERVED_STEPS,
                    self.image_size,
                    len(self.speed_experts),
                ),
                1.0 / len(self.speed_experts),
                device=wind.device,
                dtype=wind.dtype,
            )
            evidence = torch.ones(
                batch_size,
                OBSERVED_STEPS,
                self.image_size,
                device=wind.device,
                dtype=wind.dtype,
            )
            image_keep = torch.zeros_like(image_keep)

        query_hours = torch.cat([self.hindcast_hours, self.forecast_hours])
        transport, coverage, expected_delay, transport_entropy = self._transport(
            expert_probability, evidence, time_keep, query_hours
        )
        transport_hindcast = transport[:, :HINDCAST_STEPS]
        transport_forecast = transport[:, HINDCAST_STEPS:]
        forecast_coverage = coverage[:, HINDCAST_STEPS:]

        scale = self.baseline_residual_scale.to(dtype=wind.dtype).unsqueeze(0)
        difference = transport_forecast - ar_base
        horizon = self.forecast_hours.to(dtype=wind.dtype).unsqueeze(0).expand(
            batch_size, -1
        ) / self.forecast_hours[-1]
        wind_last = wind[:, -1:].expand(-1, FORECAST_STEPS)
        wind_trend = ((wind[:, -1] - wind[:, 0]) / (OBSERVED_STEPS - 1)).unsqueeze(
            1
        ).expand(-1, FORECAST_STEPS)
        normalized_coverage = torch.log1p(forecast_coverage) / 8.0
        fusion_features = torch.stack(
            [
                ar_base,
                transport_forecast,
                difference,
                difference / scale,
                horizon,
                wind_last,
                wind_trend,
                normalized_coverage,
            ],
            dim=-1,
        )
        learned_adjustment = 0.25 * torch.tanh(
            self.fusion_head(fusion_features).squeeze(-1)
        )
        image_correction = (
            self.correction_cap_multiplier
            * scale
            * torch.tanh(
                self.transport_strength * difference / scale + learned_adjustment
            )
            * image_keep.unsqueeze(-1)
        )
        prediction = ar_base + image_correction
        correction_gate = image_keep.unsqueeze(-1).expand(-1, FORECAST_STEPS)

        spatial_attention = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(
            1e-7
        )
        expert_entropy = -(
            expert_probability.clamp_min(1e-8).log() * expert_probability
        ).sum(dim=-1) / math.log(len(self.speed_experts))
        self._last_diagnostics = {
            "transport_hindcast_rms_kms": torch.sqrt(
                transport_hindcast.square().mean()
            )
            * 1000.0,
            "transport_forecast_mean_kms": transport_forecast.mean() * 1000.0,
            "transport_forecast_std_kms": transport_forecast.std() * 1000.0,
            "transport_expected_delay_h": expected_delay[:, HINDCAST_STEPS:].mean(),
            "transport_entropy": transport_entropy[:, HINDCAST_STEPS:].mean(),
            "expert_entropy": expert_entropy.mean(),
            "source_evidence_mean": evidence.mean(),
            "image_correction_rms_kms": torch.sqrt(
                image_correction.square().mean()
            )
            * 1000.0,
        }
        components = {
            "ar_base": ar_base,
            "wind_base": ar_base,
            "wind_residual": torch.zeros_like(ar_base),
            "transport_forecast": transport_forecast,
            "image_correction": image_correction,
            "correction_gate": correction_gate,
        }
        aux = {
            "transport_hindcast": transport_hindcast,
            "transport_forecast": transport_forecast,
            "hindcast_wind": wind[:, -HINDCAST_STEPS:],
            "expert_probability": expert_probability,
            "source_evidence": evidence,
            "spatial_attention": spatial_attention.unsqueeze(2),
            "time_keep": time_keep,
            "image_keep": image_keep,
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
