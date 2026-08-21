import torch

from model_solar_lstm_v12 import FORECAST_STEPS, SolarWindLagLSTMV12


def main():
    torch.manual_seed(777)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SolarWindLagLSTMV12(
        image_size=64,
        ar_coefficients=[-0.34, 1.28],
        ar_intercept=0.025,
        baseline_residual_scale=[0.08] * FORECAST_STEPS,
        grid_rows=2,
        grid_columns=8,
        cell_dim=16,
        frame_dim=64,
        lstm_hidden_dim=64,
        wind_feature_dim=32,
        dropout=0.0,
        time_mask_prob=0.0,
        modality_drop_prob=0.0,
    ).to(device)
    model.train()
    images = torch.rand(1, 20, 2, 64, 64, device=device)
    wind = 0.35 + 0.15 * torch.rand(1, 20, device=device)
    prediction, components, aux = model(
        images, wind, return_components=True, return_aux=True
    )
    assert prediction.shape == (1, FORECAST_STEPS)
    assert aux["lag_attention"].shape == (1, FORECAST_STEPS, 20)
    assert aux["spatial_attention"].shape == (1, 20, 2, 8)
    assert torch.allclose(
        aux["lag_attention"].sum(dim=-1),
        torch.ones(1, FORECAST_STEPS, device=device),
        atol=1e-5,
    )
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    model.eval()
    dropped, dropped_components = model(
        images,
        wind,
        return_components=True,
        time_keep=torch.ones(1, 20, device=device),
        image_keep=torch.zeros(1, device=device),
    )
    assert torch.equal(dropped, dropped_components["wind_base"])
    assert torch.count_nonzero(dropped_components["image_correction"]) == 0
    print(
        "v12 smoke passed: "
        f"device={device} "
        f"prediction={tuple(prediction.shape)} "
        f"lag_attention={tuple(aux['lag_attention'].shape)} "
        f"spatial_attention={tuple(aux['spatial_attention'].shape)}"
    )


if __name__ == "__main__":
    main()
