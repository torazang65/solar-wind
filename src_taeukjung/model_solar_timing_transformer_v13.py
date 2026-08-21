import math

import torch
from torch import nn
from torch.nn import functional as F

from model_solar_lstm_unet_v12_1 import LiteUNetTokenEncoder
from model_solar_source_map_v11 import make_soft_solar_disk_mask
from model_solar_source_map_v11_2 import MPSCompatibleGridPool


OBSERVED_STEPS = 20
HINDCAST_STEPS = 13
FORECAST_STEPS = 12
QUERY_STEPS = HINDCAST_STEPS + FORECAST_STEPS
ARCHITECTURE_NAME = "SolarWindTimingTransformerV13"
FILE_STEM = "solar_timing_transformer_v13"


def _logit(probability):
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


class PhysicsCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, feedforward_dim, dropout):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by attention heads")
        self.heads = int(heads)
        self.head_dim = int(d_model // heads)
        self.dropout = float(dropout)
        self.query_self_norm = nn.LayerNorm(d_model)
        self.query_self_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(d_model)
        self.source_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def _split_heads(self, features):
        batch_size, steps, _ = features.shape
        return features.view(
            batch_size, steps, self.heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, queries, sources, physical_bias, valid_source):
        normalized_queries = self.query_self_norm(queries)
        self_context, _ = self.query_self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self_context

        query = self._split_heads(
            self.query_projection(self.cross_query_norm(queries))
        )
        normalized_sources = self.source_norm(sources)
        key = self._split_heads(self.key_projection(normalized_sources))
        value = self._split_heads(self.value_projection(normalized_sources))
        logits = torch.einsum("bhqd,bhnd->bhqn", query, key)
        logits = logits / math.sqrt(self.head_dim)
        logits = logits + physical_bias.unsqueeze(1)
        logits = logits.masked_fill(~valid_source.unsqueeze(1), -1e4)
        attention = torch.softmax(logits, dim=-1)
        context_attention = F.dropout(
            attention, p=self.dropout, training=self.training
        )
        context = torch.einsum("bhqn,bhnd->bhqd", context_attention, value)
        context = context.transpose(1, 2).contiguous().flatten(2)
        queries = queries + self.output_projection(context)
        queries = queries + self.feedforward(self.feedforward_norm(queries))
        return queries, attention


class SolarWindTimingTransformerV13(nn.Module):
    """U-Net source tokens with speed-locked physical timing attention."""

    def __init__(
        self,
        image_size=64,
        use_images=True,
        ar_coefficients=None,
        ar_intercept=0.0,
        baseline_residual_scale=None,
        grid_rows=2,
        grid_columns=8,
        unet_channels=(12, 16, 24, 40, 56),
        d_model=96,
        attention_heads=4,
        decoder_layers=2,
        feedforward_dim=192,
        dropout=0.15,
        time_mask_prob=0.15,
        modality_drop_prob=0.25,
        delta_gain=1.0,
        timing_sigma_hours=18.0,
        physical_prior_min_strength=1.0,
        physical_prior_max_strength=4.0,
        physical_prior_init_strength=2.0,
        maximum_blend=0.50,
        initial_blend=0.05,
        correction_cap_multiplier=1.0,
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
        if coefficients.ndim != 1 or not 1 <= len(coefficients) <= OBSERVED_STEPS:
            raise ValueError("invalid AR coefficients")
        if residual_scale.shape != (FORECAST_STEPS,) or torch.any(
            residual_scale <= 0
        ):
            raise ValueError("baseline_residual_scale must contain 12 positives")
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")
        if grid_rows <= 0 or grid_columns <= 0:
            raise ValueError("grid dimensions must be positive")
        if d_model <= 0 or feedforward_dim <= 0 or decoder_layers <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if not 0.0 <= time_mask_prob < 1.0:
            raise ValueError("time_mask_prob must be in [0, 1)")
        if not 0.0 <= modality_drop_prob < 1.0:
            raise ValueError("modality_drop_prob must be in [0, 1)")
        if delta_gain <= 0.0 or timing_sigma_hours <= 0.0:
            raise ValueError("delta_gain and timing_sigma_hours must be positive")
        if not 0.0 <= physical_prior_min_strength < physical_prior_max_strength:
            raise ValueError("invalid physical prior strength interval")
        if not (
            physical_prior_min_strength
            <= physical_prior_init_strength
            <= physical_prior_max_strength
        ):
            raise ValueError("physical prior initialization is outside its range")
        if not 0.0 < initial_blend <= maximum_blend <= 1.0:
            raise ValueError("blend strengths must satisfy 0 < initial <= max <= 1")
        if correction_cap_multiplier <= 0.0:
            raise ValueError("correction_cap_multiplier must be positive")

        self.image_size = int(image_size)
        self.use_images = bool(use_images)
        self.grid_rows = int(grid_rows)
        self.grid_columns = int(grid_columns)
        self.unet_channels = tuple(int(value) for value in unet_channels)
        self.d_model = int(d_model)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.feedforward_dim = int(feedforward_dim)
        self.time_mask_prob = float(time_mask_prob)
        self.modality_drop_prob = float(modality_drop_prob)
        self.delta_gain = float(delta_gain)
        self.timing_sigma_hours = float(timing_sigma_hours)
        self.physical_prior_min_strength = float(
            physical_prior_min_strength
        )
        self.physical_prior_max_strength = float(
            physical_prior_max_strength
        )
        self.maximum_blend = float(maximum_blend)
        self.correction_cap_multiplier = float(correction_cap_multiplier)
        self.apply_solar_disk_mask = bool(apply_solar_disk_mask)
        self.omega_deg_per_hour = 360.0 / (27.2753 * 24.0)

        self.register_buffer("ar_coefficients", coefficients)
        self.register_buffer("ar_intercept", torch.tensor(float(ar_intercept)))
        self.register_buffer("baseline_residual_scale", residual_scale)
        self.register_buffer(
            "image_age_hours",
            torch.arange(
                OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32
            )
            * 6.0,
        )
        self.register_buffer(
            "hindcast_hours", torch.arange(-12, 1, dtype=torch.float32) * 6.0
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
            -90.0 + 90.0 / self.grid_columns,
            90.0 - 90.0 / self.grid_columns,
            self.grid_columns,
        )
        lat_grid, lon_grid = torch.meshgrid(latitude, longitude, indexing="ij")
        self.register_buffer("cell_latitude", lat_grid)
        self.register_buffer("cell_longitude_deg", lon_grid)

        self.unet_encoder = LiteUNetTokenEncoder(
            channels=self.unet_channels,
            output_channels=128,
        )
        self.spatial_pool = MPSCompatibleGridPool(
            self.grid_rows, self.grid_columns
        )
        self.token_projection = nn.Linear(128, self.d_model)
        self.coordinate_projection = nn.Linear(3, self.d_model, bias=False)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.source_speed_head = nn.Linear(self.d_model, 1)
        self.source_evidence_head = nn.Linear(self.d_model, 1)

        self.query_embedding = nn.Parameter(
            torch.empty(QUERY_STEPS, self.d_model)
        )
        self.query_time_projection = nn.Linear(3, self.d_model, bias=False)
        self.query_blocks = nn.ModuleList(
            [
                PhysicsCrossAttentionBlock(
                    self.d_model,
                    self.attention_heads,
                    self.feedforward_dim,
                    dropout,
                )
                for _ in range(self.decoder_layers)
            ]
        )
        prior_fraction = (
            physical_prior_init_strength - self.physical_prior_min_strength
        ) / (
            self.physical_prior_max_strength
            - self.physical_prior_min_strength
        )
        self.physical_prior_strength_raw = nn.Parameter(
            torch.tensor(_logit(prior_fraction))
        )
        self.effective_distance_raw = nn.Parameter(torch.tensor(-0.144))
        blend_fraction = initial_blend / self.maximum_blend
        self.blend_strength_raw = nn.Parameter(
            torch.full((FORECAST_STEPS,), _logit(blend_fraction))
        )

        nn.init.normal_(self.query_embedding, std=0.02)
        nn.init.normal_(self.source_speed_head.weight, std=0.02)
        nn.init.constant_(self.source_speed_head.bias, -0.96)
        nn.init.normal_(self.source_evidence_head.weight, std=0.01)
        nn.init.zeros_(self.source_evidence_head.bias)
        self._last_diagnostics = {}

    @property
    def query_hours(self):
        return torch.cat([self.hindcast_hours, self.horizon_hours])

    def effective_distance(self):
        return 30.0 + 25.0 * torch.sigmoid(self.effective_distance_raw)

    def physical_prior_strength(self):
        interval = (
            self.physical_prior_max_strength
            - self.physical_prior_min_strength
        )
        return self.physical_prior_min_strength + interval * torch.sigmoid(
            self.physical_prior_strength_raw
        )

    def blend_strength(self):
        return self.maximum_blend * torch.sigmoid(self.blend_strength_raw)

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
            time_keep[:, 0] = 1.0
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

    def _encode_sources(self, images):
        batch_size, time_steps = images.shape[:2]
        if time_steps != OBSERVED_STEPS:
            raise ValueError(f"expected {OBSERVED_STEPS} image steps")
        features = self._prepare_image_channels(images).reshape(
            batch_size * time_steps, 4, self.image_size, self.image_size
        )
        features = self.unet_encoder(features)
        height, width = features.shape[-2:]
        features = features.view(
            batch_size, time_steps, 128, height, width
        ).permute(0, 2, 1, 3, 4).contiguous()
        features = self.spatial_pool(features)
        cells = features.permute(0, 2, 3, 4, 1).contiguous()

        age = self.image_age_hours.view(
            OBSERVED_STEPS, 1, 1
        ).expand(-1, self.grid_rows, self.grid_columns)
        latitude = self.cell_latitude.unsqueeze(0).expand(
            OBSERVED_STEPS, -1, -1
        )
        longitude = self.cell_longitude_deg.unsqueeze(0).expand(
            OBSERVED_STEPS, -1, -1
        )
        coordinates = torch.stack(
            [age / 120.0, latitude, longitude / 90.0], dim=-1
        ).to(dtype=cells.dtype)
        tokens = self.token_norm(
            self.token_projection(cells) + self.coordinate_projection(coordinates)
        )
        return tokens

    def _query_tokens(self, batch_size, dtype):
        hours = self.query_hours.to(dtype=dtype)
        time_features = torch.stack(
            [
                hours / 120.0,
                torch.sin(math.pi * hours / 72.0),
                torch.cos(math.pi * hours / 72.0),
            ],
            dim=-1,
        )
        queries = self.query_embedding.to(dtype=dtype) + self.query_time_projection(
            time_features
        )
        return queries.unsqueeze(0).expand(batch_size, -1, -1)

    def _timing_prior(self, source_speed, source_evidence, time_keep):
        batch_size = source_speed.shape[0]
        age = self.image_age_hours.view(1, OBSERVED_STEPS, 1, 1).to(
            dtype=source_speed.dtype
        )
        longitude = self.cell_longitude_deg.view(
            1, 1, self.grid_rows, self.grid_columns
        ).to(dtype=source_speed.dtype)
        rotation_wait = -longitude / self.omega_deg_per_hour
        transit = self.effective_distance().to(dtype=source_speed.dtype) / source_speed
        arrival = rotation_wait + transit - age
        query_hours = self.query_hours.view(1, QUERY_STEPS, 1, 1, 1).to(
            dtype=source_speed.dtype
        )
        physical_bias = -(
            query_hours - arrival.unsqueeze(1)
        ).square() / (2.0 * self.timing_sigma_hours**2)
        physical_bias = physical_bias - physical_bias.amax(
            dim=(2, 3, 4), keepdim=True
        )
        evidence_bias = source_evidence.clamp_min(1e-4).log().unsqueeze(1)
        prior_strength = self.physical_prior_strength().to(
            dtype=source_speed.dtype
        )
        physical_bias = prior_strength * physical_bias + evidence_bias

        image_time = -self.image_age_hours
        causal = image_time.view(1, 1, OBSERVED_STEPS, 1, 1) <= (
            self.query_hours.view(1, QUERY_STEPS, 1, 1, 1)
        )
        valid = causal.expand(
            batch_size,
            QUERY_STEPS,
            OBSERVED_STEPS,
            self.grid_rows,
            self.grid_columns,
        )
        valid = valid & (
            time_keep.view(batch_size, 1, OBSERVED_STEPS, 1, 1) > 0.0
        )
        return (
            physical_bias.flatten(2),
            valid.flatten(2),
            transit,
            arrival,
        )

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
        time_keep = torch.as_tensor(
            time_keep, device=wind.device, dtype=wind.dtype
        )
        image_keep = torch.as_tensor(
            image_keep, device=wind.device, dtype=wind.dtype
        ).flatten()
        if time_keep.shape != (batch_size, OBSERVED_STEPS):
            raise ValueError("time_keep must have shape (batch, 20)")
        if image_keep.shape != (batch_size,):
            raise ValueError("image_keep must have shape (batch,)")

        if self.use_images:
            source_tokens = self._encode_sources(images)
        else:
            source_tokens = torch.zeros(
                batch_size,
                OBSERVED_STEPS,
                self.grid_rows,
                self.grid_columns,
                self.d_model,
                device=wind.device,
                dtype=wind.dtype,
            )
            image_keep = torch.zeros_like(image_keep)
        source_tokens = source_tokens * time_keep.view(
            batch_size, OBSERVED_STEPS, 1, 1, 1
        )
        source_speed = 0.25 + 0.65 * torch.sigmoid(
            self.source_speed_head(source_tokens)
        ).squeeze(-1)
        source_evidence = torch.sigmoid(
            self.source_evidence_head(source_tokens)
        ).squeeze(-1)
        physical_bias, valid_source, transit, arrival = self._timing_prior(
            source_speed, source_evidence, time_keep
        )
        flat_sources = source_tokens.flatten(1, 3)
        queries = self._query_tokens(batch_size, source_tokens.dtype)
        attention = None
        for block in self.query_blocks:
            queries, attention = block(
                queries, flat_sources, physical_bias, valid_source
            )
        timing_attention = attention.mean(dim=1)
        flat_speed = source_speed.flatten(1)
        flat_evidence = source_evidence.flatten(1)
        source_prediction = torch.einsum(
            "bqn,bn->bq", timing_attention, flat_speed
        )
        query_evidence = torch.einsum(
            "bqn,bn->bq", timing_attention, flat_evidence
        )
        hindcast = source_prediction[:, :HINDCAST_STEPS]
        source_future = source_prediction[:, HINDCAST_STEPS:]
        future_evidence = query_evidence[:, HINDCAST_STEPS:]

        correction_cap = (
            self.correction_cap_multiplier
            * self.baseline_residual_scale.to(dtype=wind.dtype)
        )
        bounded_difference = correction_cap * torch.tanh(
            (source_future - ar_base) / correction_cap
        )
        correction_gate = (
            self.blend_strength().to(dtype=wind.dtype).unsqueeze(0)
            * future_evidence
            * image_keep.unsqueeze(-1)
        )
        image_correction = correction_gate * bounded_difference
        prediction = ar_base + image_correction

        future_attention = timing_attention[:, HINDCAST_STEPS:]
        flat_age = self.image_age_hours.view(
            OBSERVED_STEPS, 1, 1
        ).expand(-1, self.grid_rows, self.grid_columns).flatten()
        attention_age = torch.einsum(
            "bhn,n->bh", future_attention, flat_age.to(dtype=wind.dtype)
        )
        attention_entropy = -(
            timing_attention.clamp_min(1e-8).log() * timing_attention
        ).sum(dim=-1) / math.log(timing_attention.shape[-1])
        self._last_diagnostics = {
            "attention_delay_h": (
                attention_age + self.horizon_hours.to(dtype=wind.dtype)
            ).mean(),
            "attention_entropy": attention_entropy.mean(),
            "source_speed_mean_kms": source_speed.mean() * 1000.0,
            "source_speed_std_kms": source_speed.std() * 1000.0,
            "source_evidence": source_evidence.mean(),
            "source_transit_h": transit.mean(),
            "physical_prior_strength": self.physical_prior_strength(),
            "effective_distance_h": self.effective_distance(),
            "correction_gate": correction_gate.mean(),
            "blend_strength": self.blend_strength().mean(),
            "image_correction_rms_kms": torch.sqrt(
                image_correction.square().mean()
            )
            * 1000.0,
        }
        components = {
            "ar_base": ar_base,
            "wind_base": ar_base,
            "source_future": source_future,
            "image_correction": image_correction,
            "correction_gate": correction_gate,
        }
        aux = {
            "hindcast": hindcast,
            "timing_attention": timing_attention,
            "source_speed": source_speed,
            "source_evidence": source_evidence,
            "transit": transit,
            "arrival": arrival,
            "time_keep": time_keep,
            "image_keep": image_keep,
            "valid_source": valid_source,
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
