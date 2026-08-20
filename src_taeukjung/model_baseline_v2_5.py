import math

from torch import nn

from config import IMAGE_SIZE
from model_baseline_v2_4 import SolarWindFixedLagMagnitudeTransformerV24


class SolarWindDeepFixedLagTransformerV25(
    SolarWindFixedLagMagnitudeTransformerV24
):
    """Higher-capacity fixed-lag model with stronger neural corrections."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        image_scale_hidden_dim=96,
        initial_image_gate=0.40,
        image_head_init_std=1e-3,
        wind_head_init_std=2e-4,
        **kwargs,
    ):
        super().__init__(image_size=image_size, **kwargs)
        if image_scale_hidden_dim <= 0:
            raise ValueError("image scale hidden dimension must be positive")
        if not 0.0 < initial_image_gate < 1.0:
            raise ValueError("initial image gate must be between zero and one")
        if image_head_init_std <= 0.0 or wind_head_init_std <= 0.0:
            raise ValueError("neural head initialization must be positive")

        self.image_scale_hidden_dim = int(image_scale_hidden_dim)
        self.initial_image_gate = float(initial_image_gate)
        self.image_head_init_std = float(image_head_init_std)
        self.wind_head_init_std = float(wind_head_init_std)

        d_model = int(kwargs.get("d_model", 128))
        dropout = float(kwargs.get("dropout", 0.15))
        self.image_scale_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.image_scale_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.image_scale_hidden_dim, 1),
        )
        nn.init.normal_(
            self.image_scale_head[-1].weight,
            std=self.image_head_init_std,
        )
        nn.init.zeros_(self.image_scale_head[-1].bias)

        nn.init.normal_(
            self.wind_residual_head.network[-1].weight,
            std=self.wind_head_init_std,
        )
        nn.init.zeros_(self.wind_residual_head.network[-1].bias)

        nn.init.normal_(self.image_gate_head.weight, std=1e-3)
        gate_logit = math.log(
            self.initial_image_gate / (1.0 - self.initial_image_gate)
        )
        nn.init.constant_(self.image_gate_head.bias, gate_logit)
