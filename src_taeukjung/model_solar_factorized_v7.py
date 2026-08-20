import torch
from torch import nn

from model_solar_cnn_v6 import SolarWindCNNTransformerV6


class SharedAxisTransformerBlock(nn.Module):
    """Apply one shared attention kernel along longitude and then time."""

    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        self.longitude_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _attend(self, sequence, norm):
        normalized = norm(sequence)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        return sequence + self.dropout(attended)

    def forward(self, memory):
        batch, steps, latitude_bins, longitude_bins, d_model = memory.shape

        longitude = memory.reshape(
            batch * steps * latitude_bins, longitude_bins, d_model
        )
        longitude = self._attend(longitude, self.longitude_norm)
        memory = longitude.reshape(
            batch, steps, latitude_bins, longitude_bins, d_model
        )

        temporal = memory.permute(0, 2, 3, 1, 4).reshape(
            batch * latitude_bins * longitude_bins, steps, d_model
        )
        temporal = self._attend(temporal, self.temporal_norm)
        memory = temporal.reshape(
            batch, latitude_bins, longitude_bins, steps, d_model
        ).permute(0, 3, 1, 2, 4)

        normalized = self.ffn_norm(memory)
        return memory + self.dropout(self.ffn(normalized))


class SolarWindFactorizedTransformerV7(SolarWindCNNTransformerV6):
    """V6 with shared-weight factorized longitude and temporal attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", 96)
        nhead = kwargs.get("nhead", 8)
        encoder_layers = kwargs.get("encoder_layers", 1)
        ff_dim = kwargs.get("ff_dim", 192)
        dropout = kwargs.get("dropout", 0.25)

        del self.longitude_time_encoder
        self.factorized_blocks = nn.ModuleList(
            [
                SharedAxisTransformerBlock(d_model, nhead, ff_dim, dropout)
                for _ in range(encoder_layers)
            ]
        )
        self.factorized_output_norm = nn.LayerNorm(d_model)

    def encode_memory(self, images, wind_tokens):
        batch = images.size(0)
        if self.use_images:
            image_tokens = self.image_encoder(images)
        else:
            image_tokens = wind_tokens.new_zeros(
                batch,
                wind_tokens.size(1),
                self.cell_count,
                self.image_token_dim,
            )

        repeated_wind = wind_tokens.unsqueeze(2).expand(-1, -1, self.cell_count, -1)
        memory = torch.cat([image_tokens, repeated_wind], dim=-1).reshape(
            batch,
            image_tokens.size(1),
            self.latitude_bins,
            self.longitude_bins,
            -1,
        )
        memory = self.memory_norm(
            memory
            + self.observed_position
            + self.latitude_position
            + self.longitude_position
        )
        for block in self.factorized_blocks:
            memory = block(memory)
        memory = self.factorized_output_norm(memory)
        return memory.reshape(batch, wind_tokens.size(1) * self.cell_count, -1)

    def encoder_attention_score_count(self):
        steps = self.observed_position.size(1)
        longitude_scores = steps * self.latitude_bins * self.longitude_bins**2
        temporal_scores = (
            self.latitude_bins * self.longitude_bins * steps**2
        )
        return len(self.factorized_blocks) * (longitude_scores + temporal_scores)
