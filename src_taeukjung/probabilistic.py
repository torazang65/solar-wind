import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankStudentTHead(nn.Module):
    """Joint heavy-tailed forecast distribution with a low-rank scale matrix."""

    def __init__(
        self,
        feature_dim,
        forecast_steps,
        rank=3,
        baseline_residual_scale=None,
        dropout=0.1,
    ):
        super().__init__()
        if baseline_residual_scale is None:
            baseline_residual_scale = torch.ones(forecast_steps)
        if len(baseline_residual_scale) != forecast_steps:
            raise ValueError(
                f"baseline_residual_scale must have {forecast_steps} values"
            )

        hidden_dim = feature_dim // 2
        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.mean_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.mean_head[-1].bias)

        self.log_scale_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.zeros_(self.log_scale_head.bias)

        self.factor_head = nn.Linear(feature_dim, rank)
        nn.init.normal_(self.factor_head.weight, std=1e-3)
        nn.init.zeros_(self.factor_head.bias)

        initial_df_minus_two = torch.tensor(3.0)
        self.raw_df = nn.Parameter(torch.log(torch.expm1(initial_df_minus_two)))
        self.register_buffer(
            "baseline_residual_scale",
            torch.as_tensor(baseline_residual_scale, dtype=torch.float32).view(
                1, -1
            ),
        )

    def forward(self, features):
        residual_mean = self.mean_head(features).squeeze(-1)
        log_scale = self.log_scale_head(features).squeeze(-1).clamp(-2.5, 2.5)
        diagonal_scale = self.baseline_residual_scale * torch.exp(log_scale)
        factors = self.factor_head(features)
        factors = 0.1 * self.baseline_residual_scale.unsqueeze(-1) * torch.tanh(factors)
        degrees_of_freedom = 2.0 + F.softplus(self.raw_df)
        return residual_mean, diagonal_scale, factors, degrees_of_freedom


def multivariate_student_t_nll(
    target_km_s,
    mean_km_s,
    diagonal_scale,
    factors,
    degrees_of_freedom,
):
    """Negative log likelihood for a low-rank multivariate Student-t."""
    target_km_s = target_km_s.float()
    mean_km_s = mean_km_s.float()
    diagonal_scale = diagonal_scale.float().clamp_min(1e-3)
    factors = factors.float()
    degrees_of_freedom = degrees_of_freedom.float().clamp_min(2.01)

    dimensions = target_km_s.shape[-1]
    scale_matrix = torch.diag_embed(diagonal_scale.square())
    scale_matrix = scale_matrix + factors @ factors.transpose(-1, -2)
    identity = torch.eye(
        dimensions, device=scale_matrix.device, dtype=scale_matrix.dtype
    )
    cholesky = torch.linalg.cholesky(scale_matrix + identity * 1e-4)

    difference = (target_km_s - mean_km_s).unsqueeze(-1)
    solved = torch.cholesky_solve(difference, cholesky)
    mahalanobis = (difference.transpose(-1, -2) @ solved).flatten()
    log_determinant = 2.0 * torch.log(
        torch.diagonal(cholesky, dim1=-2, dim2=-1)
    ).sum(dim=-1)

    log_normalizer = (
        torch.lgamma((degrees_of_freedom + dimensions) / 2.0)
        - torch.lgamma(degrees_of_freedom / 2.0)
        - 0.5
        * (
            dimensions * torch.log(degrees_of_freedom * math.pi)
            + log_determinant
        )
    )
    log_density = log_normalizer - 0.5 * (
        degrees_of_freedom + dimensions
    ) * torch.log1p(mahalanobis / degrees_of_freedom)
    return -log_density.mean()


def marginal_standard_deviation(diagonal_scale, factors, degrees_of_freedom):
    marginal_scale = torch.sqrt(
        diagonal_scale.square() + factors.square().sum(dim=-1)
    )
    variance_multiplier = degrees_of_freedom / (degrees_of_freedom - 2.0)
    return marginal_scale * torch.sqrt(variance_multiplier)
