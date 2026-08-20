import torch

from model_solar_hybrid_v10 import SolarWindAnchoredHybridV10


ARCHITECTURE_NAME = "SolarWindSelectiveHybridV101"
FILE_STEM = "solar_hybrid_v10_1"


class SolarWindSelectiveHybridV101(SolarWindAnchoredHybridV10):
    """V10 with a direct AR anchor and a regime-selective correction gate."""

    def __init__(
        self,
        wind_residual_mix=0.0,
        correction_min_gate=0.15,
        correction_surge_power=1.0,
        fast_wind_threshold_kms=550.0,
        fast_wind_scale_kms=50.0,
        fast_quiet_suppression=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= wind_residual_mix <= 1.0:
            raise ValueError("wind_residual_mix must be between zero and one")
        if not 0.0 <= correction_min_gate <= 1.0:
            raise ValueError("correction_min_gate must be between zero and one")
        if correction_surge_power <= 0.0:
            raise ValueError("correction_surge_power must be positive")
        if fast_wind_scale_kms <= 0.0:
            raise ValueError("fast_wind_scale_kms must be positive")
        if not 0.0 <= fast_quiet_suppression <= 1.0:
            raise ValueError("fast_quiet_suppression must be between zero and one")

        self.wind_residual_mix = float(wind_residual_mix)
        self.correction_min_gate = float(correction_min_gate)
        self.correction_surge_power = float(correction_surge_power)
        self.fast_wind_threshold_kms = float(fast_wind_threshold_kms)
        self.fast_wind_scale_kms = float(fast_wind_scale_kms)
        self.fast_quiet_suppression = float(fast_quiet_suppression)

        if self.wind_residual_mix == 0.0:
            self.hybrid_wind_encoder.requires_grad_(False)
            self.hybrid_wind_residual_head.requires_grad_(False)

    def _correction_gate(self, wind, surge_logit, correction):
        if surge_logit is None:
            surge_probability = torch.zeros(
                wind.shape[0], 1, device=wind.device, dtype=wind.dtype
            )
        else:
            surge_probability = torch.sigmoid(surge_logit).to(dtype=wind.dtype)

        surge_support = self.correction_min_gate + (
            1.0 - self.correction_min_gate
        ) * surge_probability.pow(self.correction_surge_power)
        latest_wind_kms = wind[:, -1:] * 1000.0
        fast_probability = torch.sigmoid(
            (latest_wind_kms - self.fast_wind_threshold_kms)
            / self.fast_wind_scale_kms
        )
        fast_quiet_gate = 1.0 - self.fast_quiet_suppression * (
            fast_probability * (1.0 - surge_probability)
        )
        gate = (surge_support * fast_quiet_gate).expand_as(correction)
        return gate, surge_probability

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_aux=False,
    ):
        _, original_components, aux = super().forward(
            images,
            wind,
            return_components=True,
            return_aux=True,
        )
        raw_wind_residual = original_components["wind_residual"]
        effective_wind_residual = raw_wind_residual * self.wind_residual_mix
        ar_baseline = original_components["ar_baseline"]
        wind_prediction = ar_baseline + effective_wind_residual

        raw_correction = original_components["correction"]
        correction_gate, surge_probability = self._correction_gate(
            wind, aux["surge_logit"], raw_correction
        )
        correction = raw_correction * correction_gate
        propagation = original_components["propagation_residual"]
        prediction = wind_prediction + propagation + correction

        previous_diagnostics = dict(self._last_hybrid_diagnostics)
        previous_diagnostics.update(
            {
                "wind_residual_rms_kms": (
                    effective_wind_residual.detach().float().square().mean().sqrt()
                    * 1000.0
                ),
                "raw_correction_rms_kms": (
                    raw_correction.detach().float().square().mean().sqrt() * 1000.0
                ),
                "correction_rms_kms": (
                    correction.detach().float().square().mean().sqrt() * 1000.0
                ),
                "correction_gate": correction_gate.detach().mean(),
                "surge_probability": surge_probability.detach().mean(),
            }
        )
        self._last_hybrid_diagnostics = previous_diagnostics

        components = {
            **original_components,
            "ar_baseline": ar_baseline,
            "raw_wind_residual": raw_wind_residual,
            "wind_residual": effective_wind_residual,
            "wind_prediction": wind_prediction,
            "raw_correction": raw_correction,
            "correction": correction,
            "correction_gate": correction_gate,
            "surge_probability": surge_probability.expand_as(correction),
        }
        aux = {**aux, **components}
        if return_components and return_aux:
            return prediction, components, aux
        if return_components:
            return prediction, components
        if return_aux:
            return prediction, aux
        return prediction
