import math

import torch
from torch import nn
from torch.nn import functional as F

from model_solar_timing_transformer_v13 import (
    FORECAST_STEPS,
    HINDCAST_STEPS,
    OBSERVED_STEPS,
    QUERY_STEPS,
    SolarWindTimingTransformerV13,
)


ARCHITECTURE_NAME = "SolarWindDeformableTimingV14"
FILE_STEM = "solar_deformable_timing_v14"


class PhysicsDeformableCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        feedforward_dim,
        dropout,
        observed_steps,
        grid_rows,
        grid_columns,
        sampling_points,
        maximum_time_offset_hours,
        maximum_longitude_offset_cells,
        dense_kernel_time_frames,
        dense_kernel_longitude_cells,
    ):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by attention heads")
        earliest_causal_sources = 8 * grid_rows * grid_columns
        if not 2 <= sampling_points <= earliest_causal_sources:
            raise ValueError(
                "sampling_points must be between 2 and the sources visible "
                "to the earliest query"
            )
        if maximum_time_offset_hours < 0.0:
            raise ValueError("maximum_time_offset_hours must be nonnegative")
        if maximum_longitude_offset_cells < 0.0:
            raise ValueError(
                "maximum_longitude_offset_cells must be nonnegative"
            )
        if dense_kernel_time_frames <= 0.0:
            raise ValueError("dense_kernel_time_frames must be positive")
        if dense_kernel_longitude_cells <= 0.0:
            raise ValueError("dense_kernel_longitude_cells must be positive")

        self.heads = int(heads)
        self.head_dim = int(d_model // heads)
        self.dropout = float(dropout)
        self.observed_steps = int(observed_steps)
        self.grid_rows = int(grid_rows)
        self.grid_columns = int(grid_columns)
        self.sampling_points = int(sampling_points)
        self.maximum_time_offset_frames = float(maximum_time_offset_hours) / 6.0
        self.maximum_longitude_offset_cells = float(
            maximum_longitude_offset_cells
        )
        self.dense_kernel_time_frames = float(dense_kernel_time_frames)
        self.dense_kernel_longitude_cells = float(
            dense_kernel_longitude_cells
        )

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
        self.offset_projection = nn.Linear(
            d_model, heads * sampling_points * 2
        )
        self.key_weight = nn.Parameter(
            torch.empty(heads, d_model, self.head_dim)
        )
        self.value_weight = nn.Parameter(
            torch.empty(heads, d_model, self.head_dim)
        )
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        anchor_mask = torch.ones(sampling_points, 2)
        anchor_mask[0] = 0.0
        self.register_buffer("offset_anchor_mask", anchor_mask)

        nn.init.zeros_(self.offset_projection.weight)
        nn.init.zeros_(self.offset_projection.bias)
        for head in range(heads):
            nn.init.xavier_uniform_(self.key_weight[head])
            nn.init.xavier_uniform_(self.value_weight[head])

    @staticmethod
    def _gather_flat(volume, index):
        batch_size, _, channels = volume.shape
        flat_index = index.reshape(batch_size, -1)
        gathered = torch.gather(
            volume,
            1,
            flat_index.unsqueeze(-1).expand(-1, -1, channels),
        )
        return gathered.view(*index.shape, channels)

    def _sample_time_longitude(self, volume, sample_time, sample_row, sample_lon):
        batch_size, _, _, _, channels = volume.shape
        flat_volume = volume.reshape(batch_size, -1, channels)
        time0 = sample_time.floor().long()
        time1 = (time0 + 1).clamp_max(self.observed_steps - 1)
        lon0 = sample_lon.floor().long()
        lon1 = (lon0 + 1).clamp_max(self.grid_columns - 1)
        row = sample_row.long()
        time_weight = sample_time - time0.to(dtype=sample_time.dtype)
        lon_weight = sample_lon - lon0.to(dtype=sample_lon.dtype)

        def flat_index(time_index, longitude_index):
            return (
                time_index * self.grid_rows * self.grid_columns
                + row * self.grid_columns
                + longitude_index
            )

        value00 = self._gather_flat(flat_volume, flat_index(time0, lon0))
        value01 = self._gather_flat(flat_volume, flat_index(time0, lon1))
        value10 = self._gather_flat(flat_volume, flat_index(time1, lon0))
        value11 = self._gather_flat(flat_volume, flat_index(time1, lon1))
        weight00 = (1.0 - time_weight) * (1.0 - lon_weight)
        weight01 = (1.0 - time_weight) * lon_weight
        weight10 = time_weight * (1.0 - lon_weight)
        weight11 = time_weight * lon_weight
        return (
            value00 * weight00.unsqueeze(-1)
            + value01 * weight01.unsqueeze(-1)
            + value10 * weight10.unsqueeze(-1)
            + value11 * weight11.unsqueeze(-1)
        )

    def _dense_attention(
        self,
        sparse_attention,
        sample_time,
        sample_row,
        sample_lon,
        valid_source,
    ):
        device = sample_time.device
        dtype = sample_time.dtype
        time_grid = torch.arange(
            self.observed_steps, device=device, dtype=dtype
        ).view(self.observed_steps, 1, 1)
        row_grid = torch.arange(
            self.grid_rows, device=device, dtype=dtype
        ).view(1, self.grid_rows, 1)
        longitude_grid = torch.arange(
            self.grid_columns, device=device, dtype=dtype
        ).view(1, 1, self.grid_columns)
        time_grid = time_grid.expand(
            self.observed_steps, self.grid_rows, self.grid_columns
        ).flatten()
        row_grid = row_grid.expand(
            self.observed_steps, self.grid_rows, self.grid_columns
        ).flatten()
        longitude_grid = longitude_grid.expand(
            self.observed_steps, self.grid_rows, self.grid_columns
        ).flatten()
        squared_distance = (
            (
                (time_grid.view(1, 1, 1, 1, -1) - sample_time.unsqueeze(-1))
                / self.dense_kernel_time_frames
            ).square()
            + (
                (longitude_grid.view(1, 1, 1, 1, -1) - sample_lon.unsqueeze(-1))
                / self.dense_kernel_longitude_cells
            ).square()
            + (
                (row_grid.view(1, 1, 1, 1, -1) - sample_row.unsqueeze(-1))
                / 0.25
            ).square()
        )
        kernel = torch.exp(-0.5 * squared_distance)
        kernel = kernel * valid_source.unsqueeze(1).unsqueeze(3).to(dtype=dtype)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bhqk,bhqkn->bhqn", sparse_attention, kernel)

    def forward(
        self,
        queries,
        source_tokens,
        source_speed,
        source_evidence,
        time_keep,
        physical_bias,
        valid_source,
        query_hours,
        effective_distance,
        physical_prior_strength,
        timing_sigma_hours,
        omega_deg_per_hour,
    ):
        batch_size, query_steps, d_model = queries.shape
        normalized_queries = self.query_self_norm(queries)
        self_context, _ = self.query_self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self_context
        cross_queries = self.cross_query_norm(queries)

        reference_logits = physical_bias.masked_fill(~valid_source, -1e4)
        reference_index = reference_logits.topk(
            self.sampling_points, dim=-1
        ).indices
        cells_per_time = self.grid_rows * self.grid_columns
        reference_time = reference_index // cells_per_time
        reference_remainder = reference_index % cells_per_time
        reference_row = reference_remainder // self.grid_columns
        reference_lon = reference_remainder % self.grid_columns

        raw_offset = self.offset_projection(cross_queries).view(
            batch_size,
            query_steps,
            self.heads,
            self.sampling_points,
            2,
        ).permute(0, 2, 1, 3, 4)
        raw_offset = torch.tanh(raw_offset) * self.offset_anchor_mask.view(
            1, 1, 1, self.sampling_points, 2
        )
        time_offset = raw_offset[..., 0] * self.maximum_time_offset_frames
        longitude_offset = (
            raw_offset[..., 1] * self.maximum_longitude_offset_cells
        )
        reference_time = reference_time.unsqueeze(1).expand(
            -1, self.heads, -1, -1
        )
        reference_row = reference_row.unsqueeze(1).expand(
            -1, self.heads, -1, -1
        )
        reference_lon = reference_lon.unsqueeze(1).expand(
            -1, self.heads, -1, -1
        )
        sample_time = reference_time.to(dtype=queries.dtype) + time_offset
        sample_lon = reference_lon.to(dtype=queries.dtype) + longitude_offset
        causal_max = torch.floor((query_hours + 114.0) / 6.0).clamp(
            0.0, float(self.observed_steps - 1)
        )
        sample_time = torch.maximum(sample_time, torch.zeros_like(sample_time))
        sample_time = torch.minimum(
            sample_time,
            causal_max.view(1, 1, query_steps, 1).to(dtype=queries.dtype),
        )
        sample_lon = sample_lon.clamp(0.0, float(self.grid_columns - 1))
        sample_row = reference_row.to(dtype=queries.dtype)

        normalized_sources = self.source_norm(source_tokens)
        sampled_sources = self._sample_time_longitude(
            normalized_sources, sample_time, sample_row, sample_lon
        )
        sampled_speed = self._sample_time_longitude(
            source_speed.unsqueeze(-1), sample_time, sample_row, sample_lon
        ).squeeze(-1)
        sampled_evidence = self._sample_time_longitude(
            source_evidence.unsqueeze(-1), sample_time, sample_row, sample_lon
        ).squeeze(-1)
        keep_volume = time_keep.view(
            batch_size, self.observed_steps, 1, 1, 1
        ).expand(
            -1, -1, self.grid_rows, self.grid_columns, 1
        )
        sampled_keep = self._sample_time_longitude(
            keep_volume, sample_time, sample_row, sample_lon
        ).squeeze(-1)

        query = self.query_projection(cross_queries).view(
            batch_size, query_steps, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = torch.einsum(
            "bhqkd,hde->bhqke", sampled_sources, self.key_weight
        )
        value = torch.einsum(
            "bhqkd,hde->bhqke", sampled_sources, self.value_weight
        )
        learned_logits = torch.einsum("bhqd,bhqkd->bhqk", query, key)
        learned_logits = learned_logits / math.sqrt(self.head_dim)

        longitude_cell_width = 180.0 / self.grid_columns
        sampled_longitude_deg = (
            -90.0 + 0.5 * longitude_cell_width
            + sample_lon * longitude_cell_width
        )
        sampled_age_hours = (self.observed_steps - 1 - sample_time) * 6.0
        sampled_transit_hours = effective_distance / sampled_speed.clamp_min(0.2)
        sampled_arrival = (
            -sampled_longitude_deg / omega_deg_per_hour
            + sampled_transit_hours
            - sampled_age_hours
        )
        timing_logits = -(
            query_hours.view(1, 1, query_steps, 1) - sampled_arrival
        ).square() / (2.0 * float(timing_sigma_hours) ** 2)
        timing_logits = timing_logits - timing_logits.amax(
            dim=-1, keepdim=True
        )
        sparse_logits = (
            learned_logits
            + physical_prior_strength * timing_logits
            + sampled_evidence.clamp_min(1e-4).log()
            + sampled_keep.clamp_min(1e-4).log()
        )
        sparse_logits = sparse_logits.masked_fill(sampled_keep <= 1e-4, -1e4)
        sparse_attention = torch.softmax(sparse_logits, dim=-1)
        context_attention = F.dropout(
            sparse_attention, p=self.dropout, training=self.training
        )
        context = torch.einsum("bhqk,bhqkd->bhqd", context_attention, value)
        context = context.permute(0, 2, 1, 3).contiguous().flatten(2)
        queries = queries + self.output_projection(context)
        queries = queries + self.feedforward(self.feedforward_norm(queries))
        dense_attention = self._dense_attention(
            sparse_attention,
            sample_time,
            sample_row,
            sample_lon,
            valid_source,
        )
        return {
            "queries": queries,
            "sparse_attention": sparse_attention,
            "dense_attention": dense_attention,
            "sampled_speed": sampled_speed,
            "sampled_evidence": sampled_evidence,
            "sample_time": sample_time,
            "sample_longitude_deg": sampled_longitude_deg,
            "time_offset_hours": time_offset * 6.0,
            "longitude_offset_cells": longitude_offset,
            "reference_index": reference_index,
        }


class SolarWindDeformableTimingV14(SolarWindTimingTransformerV13):
    """V13 speed-locking with sparse physics-guided deformable attention."""

    def __init__(
        self,
        deformable_points=8,
        maximum_time_offset_hours=12.0,
        maximum_longitude_offset_cells=1.5,
        dense_kernel_time_frames=0.75,
        dense_kernel_longitude_cells=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deformable_points = int(deformable_points)
        self.maximum_time_offset_hours = float(maximum_time_offset_hours)
        self.maximum_longitude_offset_cells = float(
            maximum_longitude_offset_cells
        )
        self.dense_kernel_time_frames = float(dense_kernel_time_frames)
        self.dense_kernel_longitude_cells = float(
            dense_kernel_longitude_cells
        )
        self.query_blocks = nn.ModuleList(
            [
                PhysicsDeformableCrossAttentionBlock(
                    self.d_model,
                    self.attention_heads,
                    self.feedforward_dim,
                    kwargs.get("dropout", 0.15),
                    OBSERVED_STEPS,
                    self.grid_rows,
                    self.grid_columns,
                    self.deformable_points,
                    self.maximum_time_offset_hours,
                    self.maximum_longitude_offset_cells,
                    self.dense_kernel_time_frames,
                    self.dense_kernel_longitude_cells,
                )
                for _ in range(self.decoder_layers)
            ]
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
        queries = self._query_tokens(batch_size, source_tokens.dtype)
        deformable = None
        for block in self.query_blocks:
            deformable = block(
                queries,
                source_tokens,
                source_speed,
                source_evidence,
                time_keep,
                physical_bias,
                valid_source,
                self.query_hours.to(dtype=source_tokens.dtype),
                self.effective_distance().to(dtype=source_tokens.dtype),
                self.physical_prior_strength().to(dtype=source_tokens.dtype),
                self.timing_sigma_hours,
                self.omega_deg_per_hour,
            )
            queries = deformable["queries"]

        sparse_attention = deformable["sparse_attention"]
        timing_attention = deformable["dense_attention"].mean(dim=1)
        source_prediction = (
            sparse_attention * deformable["sampled_speed"]
        ).sum(dim=-1).mean(dim=1)
        query_evidence = (
            sparse_attention * deformable["sampled_evidence"]
        ).sum(dim=-1).mean(dim=1)
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
        sparse_entropy = -(
            sparse_attention.clamp_min(1e-8).log() * sparse_attention
        ).sum(dim=-1) / math.log(self.deformable_points)
        self._last_diagnostics = {
            "attention_delay_h": (
                attention_age + self.horizon_hours.to(dtype=wind.dtype)
            ).mean(),
            "attention_entropy": attention_entropy.mean(),
            "sparse_attention_entropy": sparse_entropy.mean(),
            "source_speed_mean_kms": source_speed.mean() * 1000.0,
            "source_speed_std_kms": source_speed.std() * 1000.0,
            "source_evidence": source_evidence.mean(),
            "source_transit_h": transit.mean(),
            "physical_prior_strength": self.physical_prior_strength(),
            "effective_distance_h": self.effective_distance(),
            "deform_time_offset_h": deformable["time_offset_hours"].abs().mean(),
            "deform_longitude_offset_cells": deformable[
                "longitude_offset_cells"
            ].abs().mean(),
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
            "query_features": queries,
            "timing_attention": timing_attention,
            "sparse_attention": sparse_attention,
            "source_speed": source_speed,
            "source_evidence": source_evidence,
            "transit": transit,
            "arrival": arrival,
            "sample_time": deformable["sample_time"],
            "sample_longitude_deg": deformable["sample_longitude_deg"],
            "time_offset_hours": deformable["time_offset_hours"],
            "longitude_offset_cells": deformable["longitude_offset_cells"],
            "reference_index": deformable["reference_index"],
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
