import math

import torch
from torch import nn
from torch.nn import functional as F

from model_solar_source_map_v11 import make_soft_solar_disk_mask
from model_solar_source_map_v11_2 import MPSCompatibleGridPool
from src_torazang65.model import Inception3D, conv_block


OBSERVED_STEPS = 20
FORECAST_STEPS = 12
ARCHITECTURE_NAME = "SolarWindLagLSTMV12"
FILE_STEM = "solar_lag_lstm_v12"


def _logit(probability):
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


class SolarWindLagLSTMV12(nn.Module):
    """Disk-aware CNN-LSTM with a soft speed-dependent lag prior."""

    def __init__(
        self,
        image_size=64,
        use_images=True,
        ar_coefficients=None,
        ar_intercept=0.0,
        baseline_residual_scale=None,
        grid_rows=2,
        grid_columns=8,
        cell_dim=48,
        frame_dim=256,
        lstm_hidden_dim=192,
        lstm_layers=1,
        wind_feature_dim=128,
        dropout=0.15,
        time_mask_prob=0.10,
        modality_drop_prob=0.15,
        delta_gain=1.0,
        lag_hours=(72.0, 84.0, 96.0, 108.0, 120.0),
        lag_sigma_hours=12.0,
        lag_prior_max_strength=2.0,
        lag_prior_init_strength=1.0,
        wind_residual_cap_multiplier=1.0,
        image_correction_cap_multiplier=2.0,
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
        lags = torch.as_tensor(lag_hours, dtype=torch.float32)
        if coefficients.ndim != 1 or not 1 <= len(coefficients) <= OBSERVED_STEPS:
            raise ValueError("invalid AR coefficients")
        if residual_scale.shape != (FORECAST_STEPS,) or torch.any(
            residual_scale <= 0
        ):
            raise ValueError("baseline_residual_scale must contain 12 positives")
        if lags.ndim != 1 or len(lags) == 0 or torch.any(lags <= 0):
            raise ValueError("lag_hours must contain positive values")
        if grid_rows <= 0 or grid_columns <= 0:
            raise ValueError("grid dimensions must be positive")
        if min(cell_dim, frame_dim, lstm_hidden_dim, wind_feature_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if not 0.0 <= time_mask_prob < 1.0:
            raise ValueError("time_mask_prob must be in [0, 1)")
        if not 0.0 <= modality_drop_prob < 1.0:
            raise ValueError("modality_drop_prob must be in [0, 1)")
        if delta_gain <= 0.0 or lag_sigma_hours <= 0.0:
            raise ValueError("delta_gain and lag_sigma_hours must be positive")
        if lag_prior_max_strength < 0.0:
            raise ValueError("lag_prior_max_strength must be nonnegative")
        if not 0.0 <= lag_prior_init_strength <= lag_prior_max_strength:
            raise ValueError("lag prior initialization is outside its range")
        if wind_residual_cap_multiplier <= 0.0:
            raise ValueError("wind_residual_cap_multiplier must be positive")
        if image_correction_cap_multiplier <= 0.0:
            raise ValueError("image_correction_cap_multiplier must be positive")

        self.image_size = int(image_size)
        self.use_images = bool(use_images)
        self.grid_rows = int(grid_rows)
        self.grid_columns = int(grid_columns)
        self.cell_dim = int(cell_dim)
        self.frame_dim = int(frame_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.wind_feature_dim = int(wind_feature_dim)
        self.time_mask_prob = float(time_mask_prob)
        self.modality_drop_prob = float(modality_drop_prob)
        self.delta_gain = float(delta_gain)
        self.lag_sigma_hours = float(lag_sigma_hours)
        self.lag_prior_max_strength = float(lag_prior_max_strength)
        self.wind_residual_cap_multiplier = float(wind_residual_cap_multiplier)
        self.image_correction_cap_multiplier = float(
            image_correction_cap_multiplier
        )
        self.apply_solar_disk_mask = bool(apply_solar_disk_mask)

        self.register_buffer("ar_coefficients", coefficients)
        self.register_buffer("ar_intercept", torch.tensor(float(ar_intercept)))
        self.register_buffer("baseline_residual_scale", residual_scale)
        self.register_buffer("lag_hours", lags)
        self.register_buffer(
            "image_age_hours",
            torch.arange(
                OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32
            )
            * 6.0,
        )
        self.register_buffer(
            "horizon_hours",
            torch.arange(1, FORECAST_STEPS + 1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "solar_disk_mask",
            make_soft_solar_disk_mask(
                image_size,
                center_fraction=solar_disk_center_fraction,
                radius_fraction=solar_disk_radius_fraction,
                edge_pixels=solar_disk_edge_pixels,
            ),
        )
        latitude = torch.linspace(
            1.0 - 1.0 / self.grid_rows,
            -1.0 + 1.0 / self.grid_rows,
            self.grid_rows,
        )
        longitude = torch.linspace(
            -1.0 + 1.0 / self.grid_columns,
            1.0 - 1.0 / self.grid_columns,
            self.grid_columns,
        )
        lat_grid, lon_grid = torch.meshgrid(latitude, longitude, indexing="ij")
        self.register_buffer(
            "cell_coordinates", torch.stack([lat_grid, lon_grid], dim=-1)
        )

        if self.lag_prior_max_strength == 0.0:
            prior_raw = 0.0
        else:
            prior_raw = _logit(
                lag_prior_init_strength / self.lag_prior_max_strength
            )
        self.lag_prior_strength_raw = nn.Parameter(torch.tensor(prior_raw))

        self.stem = nn.Sequential(
            conv_block(4, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        self.image_blocks = nn.ModuleList()
        in_channels = 32
        for block_index in range(3):
            if block_index < 2:
                pool_stride = (1, 2, 2)
            else:
                # Preserve east-west resolution for 2x8 pooling at 64 px.
                pool_stride = (1, 2, 1)
            self.image_blocks.append(
                nn.Sequential(
                    Inception3D(in_channels, 32),
                    nn.MaxPool3d(
                        kernel_size=(1, 3, 3),
                        stride=pool_stride,
                        padding=(0, 1, 1),
                    ),
                )
            )
            in_channels = 128
        self.spatial_pool = MPSCompatibleGridPool(
            self.grid_rows, self.grid_columns
        )
        self.cell_projection = nn.Linear(128, self.cell_dim)
        self.coordinate_projection = nn.Linear(2, self.cell_dim, bias=False)
        self.cell_norm = nn.LayerNorm(self.cell_dim)
        self.spatial_importance_head = nn.Linear(self.cell_dim, 1)
        frame_input_dim = self.cell_dim * (
            self.grid_rows * self.grid_columns + 1
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(frame_input_dim, self.frame_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frame_norm = nn.LayerNorm(self.frame_dim)
        self.time_embedding = nn.Parameter(
            torch.empty(OBSERVED_STEPS, self.frame_dim)
        )

        self.image_lstm = nn.LSTM(
            input_size=self.frame_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout if self.lstm_layers > 1 else 0.0,
        )
        self.lstm_output_norm = nn.LayerNorm(self.lstm_hidden_dim)
        self.attention_key = nn.Linear(
            self.lstm_hidden_dim, self.lstm_hidden_dim, bias=False
        )
        self.attention_value = nn.Linear(
            self.lstm_hidden_dim, self.lstm_hidden_dim, bias=False
        )
        self.horizon_query = nn.Parameter(
            torch.empty(FORECAST_STEPS, self.lstm_hidden_dim)
        )

        wind_input_dim = OBSERVED_STEPS + (OBSERVED_STEPS - 1) + 4
        self.wind_encoder = nn.Sequential(
            nn.Linear(wind_input_dim, self.wind_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.wind_feature_dim, self.wind_feature_dim),
            nn.GELU(),
        )
        self.wind_residual_head = nn.Linear(
            self.wind_feature_dim, FORECAST_STEPS
        )
        lag_input_dim = self.wind_feature_dim + self.lstm_hidden_dim
        self.lag_mixture_head = nn.Linear(
            lag_input_dim, FORECAST_STEPS * len(self.lag_hours)
        )
        fusion_dim = 2 * self.lstm_hidden_dim + self.wind_feature_dim
        self.image_correction_head = nn.Sequential(
            nn.Linear(fusion_dim, self.lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.lstm_hidden_dim, 1),
        )
        self.correction_gate_head = nn.Linear(fusion_dim, 1)

        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.horizon_query, std=0.02)
        nn.init.zeros_(self.spatial_importance_head.weight)
        nn.init.zeros_(self.spatial_importance_head.bias)
        nn.init.zeros_(self.wind_residual_head.weight)
        nn.init.zeros_(self.wind_residual_head.bias)
        nn.init.zeros_(self.lag_mixture_head.weight)
        nn.init.zeros_(self.lag_mixture_head.bias)
        nn.init.zeros_(self.image_correction_head[-1].weight)
        nn.init.zeros_(self.image_correction_head[-1].bias)
        nn.init.zeros_(self.correction_gate_head.weight)
        nn.init.constant_(self.correction_gate_head.bias, _logit(0.20))

        self._last_diagnostics = {}

    def lag_prior_strength(self):
        return self.lag_prior_max_strength * torch.sigmoid(
            self.lag_prior_strength_raw
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

    def _wind_features(self, wind):
        differences = wind[:, 1:] - wind[:, :-1]
        summary = torch.stack(
            [
                wind[:, -1],
                wind.mean(dim=1),
                wind.std(dim=1),
                wind[:, -1] - wind[:, 0],
            ],
            dim=1,
        )
        return self.wind_encoder(torch.cat([wind, differences, summary], dim=1))

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

    def _prepare_image_channels(self, images):
        if self.apply_solar_disk_mask:
            images = images * self.solar_disk_mask.to(dtype=images.dtype)
        differences = torch.zeros_like(images)
        differences[:, 1:] = images[:, 1:] - images[:, :-1]
        return torch.cat([images, self.delta_gain * differences], dim=2)

    def _project_frame_features(self, features, batch_size):
        """Convert a B,C,T,H,W feature map into LSTM frame tokens."""
        features = self.spatial_pool(features)
        cells = features.permute(0, 2, 3, 4, 1).contiguous()
        coordinates = self.cell_coordinates.to(dtype=cells.dtype)
        cells = self.cell_norm(
            self.cell_projection(cells) + self.coordinate_projection(coordinates)
        )
        spatial_logits = self.spatial_importance_head(cells).squeeze(-1).flatten(2)
        spatial_attention = torch.softmax(spatial_logits, dim=2)
        flattened_cells = cells.flatten(2, 3)
        pooled_cell = (
            spatial_attention.unsqueeze(-1) * flattened_cells
        ).sum(dim=2)
        frame_input = torch.cat(
            [flattened_cells.flatten(2), pooled_cell], dim=2
        )
        frame_tokens = self.frame_norm(self.frame_projection(frame_input))
        return frame_tokens, spatial_attention.view(
            batch_size, OBSERVED_STEPS, self.grid_rows, self.grid_columns
        )

    def _encode_images(self, images):
        features = self._prepare_image_channels(images).permute(
            0, 2, 1, 3, 4
        ).contiguous()
        features = self.stem(features)
        for block in self.image_blocks:
            features = block(features)
        return self._project_frame_features(features, images.shape[0])

    def _lag_attention(self, lstm_output, wind_features, time_keep):
        batch_size = lstm_output.shape[0]
        keys = self.attention_key(lstm_output)
        values = self.attention_value(lstm_output)
        query = self.horizon_query
        learned_logits = torch.einsum("btd,hd->bht", keys, query) / math.sqrt(
            self.lstm_hidden_dim
        )

        lag_input = torch.cat([wind_features, lstm_output[:, -1]], dim=1)
        lag_logits = self.lag_mixture_head(lag_input).view(
            batch_size, FORECAST_STEPS, len(self.lag_hours)
        )
        lag_mixture = torch.softmax(lag_logits, dim=-1)
        delay = self.horizon_hours.view(FORECAST_STEPS, 1) + self.image_age_hours.view(
            1, OBSERVED_STEPS
        )
        expert_log_prior = -(
            delay.view(FORECAST_STEPS, 1, OBSERVED_STEPS)
            - self.lag_hours.view(1, -1, 1)
        ).square() / (2.0 * self.lag_sigma_hours**2)
        mixed_log_prior = torch.logsumexp(
            lag_mixture.clamp_min(1e-8).log().unsqueeze(-1)
            + expert_log_prior.unsqueeze(0),
            dim=2,
        )
        mixed_log_prior = mixed_log_prior - mixed_log_prior.amax(
            dim=-1, keepdim=True
        )
        logits = learned_logits + self.lag_prior_strength() * mixed_log_prior
        logits = logits.masked_fill(time_keep.unsqueeze(1) <= 0.0, -1e4)
        attention = torch.softmax(logits, dim=-1)
        context = torch.einsum("bht,btd->bhd", attention, values)
        return context, attention, lag_mixture

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
        wind_features = self._wind_features(wind)
        wind_residual = (
            self.wind_residual_cap_multiplier
            * self.baseline_residual_scale.to(dtype=wind.dtype)
            * torch.tanh(self.wind_residual_head(wind_features))
        )
        wind_base = ar_base + wind_residual

        if time_keep is None or image_keep is None:
            sampled_time_keep, sampled_image_keep = self._sample_augmentation(
                batch_size, wind.device, wind.dtype
            )
            if time_keep is None:
                time_keep = sampled_time_keep
            if image_keep is None:
                image_keep = sampled_image_keep
        time_keep = torch.as_tensor(
            time_keep, device=wind.device, dtype=wind.dtype
        )
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
            frame_tokens, spatial_attention = self._encode_images(images)
        else:
            frame_tokens = torch.zeros(
                batch_size,
                OBSERVED_STEPS,
                self.frame_dim,
                device=wind.device,
                dtype=wind.dtype,
            )
            spatial_attention = torch.full(
                (
                    batch_size,
                    OBSERVED_STEPS,
                    self.grid_rows,
                    self.grid_columns,
                ),
                1.0 / (self.grid_rows * self.grid_columns),
                device=wind.device,
                dtype=wind.dtype,
            )
            image_keep = torch.zeros_like(image_keep)
        frame_tokens = frame_tokens + self.time_embedding.to(
            dtype=frame_tokens.dtype
        )
        frame_tokens = frame_tokens * time_keep.unsqueeze(-1)
        frame_tokens = frame_tokens * image_keep.view(batch_size, 1, 1)
        lstm_output, _ = self.image_lstm(frame_tokens)
        lstm_output = self.lstm_output_norm(lstm_output)
        context, lag_attention, lag_mixture = self._lag_attention(
            lstm_output, wind_features, time_keep
        )

        queries = self.horizon_query.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_wind = wind_features.unsqueeze(1).expand(
            -1, FORECAST_STEPS, -1
        )
        fusion = torch.cat([context, queries, expanded_wind], dim=-1)
        ungated_correction = (
            self.image_correction_cap_multiplier
            * self.baseline_residual_scale.to(dtype=wind.dtype)
            * torch.tanh(self.image_correction_head(fusion).squeeze(-1))
        )
        correction_gate = torch.sigmoid(
            self.correction_gate_head(fusion).squeeze(-1)
        ) * image_keep.unsqueeze(-1)
        image_correction = correction_gate * ungated_correction
        prediction = wind_base + image_correction

        attention_age = (
            lag_attention * self.image_age_hours.to(dtype=wind.dtype)
        ).sum(dim=-1)
        expected_lag = (
            lag_mixture * self.lag_hours.to(dtype=wind.dtype)
        ).sum(dim=-1)
        attention_entropy = -(
            lag_attention.clamp_min(1e-8).log() * lag_attention
        ).sum(dim=-1) / math.log(OBSERVED_STEPS)
        spatial_flat = spatial_attention.flatten(2)
        spatial_entropy = -(
            spatial_flat.clamp_min(1e-8).log() * spatial_flat
        ).sum(dim=-1) / math.log(self.grid_rows * self.grid_columns)
        self._last_diagnostics = {
            "attention_age_h": attention_age.mean(),
            "attention_delay_h": (
                attention_age + self.horizon_hours.to(dtype=wind.dtype)
            ).mean(),
            "attention_entropy": attention_entropy.mean(),
            "spatial_attention_entropy": spatial_entropy.mean(),
            "expected_lag_h": expected_lag.mean(),
            "lag_prior_strength": self.lag_prior_strength(),
            "correction_gate": correction_gate.mean(),
            "wind_residual_rms_kms": torch.sqrt(wind_residual.square().mean())
            * 1000.0,
            "image_correction_rms_kms": torch.sqrt(
                image_correction.square().mean()
            )
            * 1000.0,
        }

        components = {
            "ar_base": ar_base,
            "wind_base": wind_base,
            "wind_residual": wind_residual,
            "image_correction": image_correction,
            "correction_gate": correction_gate,
        }
        aux = {
            "lag_attention": lag_attention,
            "lag_mixture": lag_mixture,
            "spatial_attention": spatial_attention,
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
