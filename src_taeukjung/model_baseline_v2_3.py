import torch
from torch import nn

from config import IMAGE_SIZE
from model_baseline_v2_2 import SolarWindBaselineSpatialTransformerV22
from model_solar_probabilistic import FORECAST_STEPS, OBSERVED_STEPS


class LevelDifferenceWindEncoder(nn.Module):
    """Encode absolute speed, relative history, and first differences."""

    def __init__(self, output_dim, dropout):
        super().__init__()
        feature_dim = OBSERVED_STEPS + (OBSERVED_STEPS - 1) + 1
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.SELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, output_dim),
            nn.SELU(inplace=True),
        )

    def forward(self, wind):
        latest = wind[:, -1:]
        relative_history = wind - latest
        innovations = wind[:, 1:] - wind[:, :-1]
        features = torch.cat([latest, relative_history, innovations], dim=1)
        return self.network(features)


class BoundedWindResidualHead(nn.Module):
    def __init__(self, input_dim, residual_scale, cap_multiplier, dropout):
        super().__init__()
        if cap_multiplier <= 0.0:
            raise ValueError("wind residual cap multiplier must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, FORECAST_STEPS),
        )
        cap = torch.as_tensor(residual_scale, dtype=torch.float32)
        self.register_buffer("cap", cap.view(1, -1) * float(cap_multiplier))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        raw = self.network(features)
        cap = self.cap.to(dtype=raw.dtype)
        return cap * torch.tanh(raw / cap.clamp_min(1e-6))


class SolarWindARNeuralTransformerV23(SolarWindBaselineSpatialTransformerV22):
    """Global AR wind forecast plus bounded neural and image residuals."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        ar_coefficients=None,
        ar_intercept=0.0,
        ar_ridge_strength=30.0,
        wind_residual_cap_multiplier=0.75,
        **kwargs,
    ):
        super().__init__(image_size=image_size, **kwargs)
        if ar_coefficients is None:
            ar_coefficients = [0.0, 1.0]
        coefficients = torch.as_tensor(ar_coefficients, dtype=torch.float32)
        if coefficients.ndim != 1 or not 1 <= len(coefficients) <= OBSERVED_STEPS:
            raise ValueError("invalid AR coefficients")
        self.register_buffer("ar_coefficients", coefficients)
        self.register_buffer(
            "ar_intercept", torch.tensor(float(ar_intercept), dtype=torch.float32)
        )
        self.ar_order = int(len(coefficients))
        self.ar_ridge_strength = float(ar_ridge_strength)
        self.wind_residual_cap_multiplier = float(wind_residual_cap_multiplier)

        dropout = float(kwargs.get("dropout", 0.20))
        wind_dim = int(kwargs.get("wind_dim", 64))
        self.wind_encoder = LevelDifferenceWindEncoder(wind_dim, dropout)
        self.wind_residual_head = BoundedWindResidualHead(
            wind_dim,
            self.baseline_residual_scale.reshape(-1),
            self.wind_residual_cap_multiplier,
            dropout,
        )

    def linear_baseline(self, wind):
        history = [wind[:, index] for index in range(wind.shape[1])]
        predictions = []
        coefficients = self.ar_coefficients.to(dtype=wind.dtype)
        intercept = self.ar_intercept.to(dtype=wind.dtype)
        for _ in range(FORECAST_STEPS):
            context = torch.stack(history[-self.ar_order :], dim=1)
            next_value = intercept + context @ coefficients
            history.append(next_value)
            predictions.append(next_value)
        return torch.stack(predictions, dim=1)
