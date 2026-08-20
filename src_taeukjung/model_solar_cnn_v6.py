import torch

from model_solar_geometry_v3 import SolarGeometryEncoderV3
from model_solar_physics_v5 import SolarWindPhysicsTransformerV5


class SolarWindCNNTransformerV6(SolarWindPhysicsTransformerV5):
    """V5 forecast path with the learnable V3 CEA CNN restored."""

    def __init__(self, *args, visual_dropout=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        image_size = kwargs.get(
            "image_size", self.image_features.reprojection.sampling_grid.shape[1]
        )
        mask_radius = kwargs.get("solar_disk_radius_fraction", 0.49)
        cea_radius = kwargs.get("solar_cea_radius_fraction", 0.42)
        d_model = kwargs.get("d_model", 96)
        wind_dim = kwargs.get("wind_dim", 24)
        image_dim = d_model - wind_dim

        del self.image_features
        del self.image_projection
        self.image_encoder = SolarGeometryEncoderV3(
            image_size,
            mask_radius,
            cea_radius,
            image_dim,
            spatial_height=self.latitude_bins,
            spatial_width=self.longitude_bins,
            visual_dropout=visual_dropout,
        )
        self.image_token_dim = image_dim

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

        axial = memory.permute(0, 2, 1, 3, 4).reshape(
            batch * self.latitude_bins,
            wind_tokens.size(1) * self.longitude_bins,
            -1,
        )
        axial = self.longitude_time_encoder(axial)
        memory = axial.reshape(
            batch,
            self.latitude_bins,
            wind_tokens.size(1),
            self.longitude_bins,
            -1,
        )
        return memory.permute(0, 2, 1, 3, 4).reshape(
            batch, wind_tokens.size(1) * self.cell_count, -1
        )
