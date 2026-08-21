import torch

from model_solar_native_profile_lstm_v16 import SolarWindNativeProfileLSTMV16


def main():
    torch.manual_seed(777)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SolarWindNativeProfileLSTMV16(
        image_size=64,
        ar_coefficients=[-0.34, 1.28],
        ar_intercept=0.025,
        baseline_residual_scale=[0.08] * 12,
        column_dim=12,
        frame_dim=64,
        lstm_hidden_dim=48,
        wind_feature_dim=32,
        lag_hours=(96.0,),
        dropout=0.0,
        time_mask_prob=0.0,
        modality_drop_prob=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = torch.rand(2, 20, 2, 64, 64, device=device)
    wind = 0.35 + 0.15 * torch.rand(2, 20, device=device)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction, components, aux = model(
            images, wind, return_components=True, return_aux=True
        )
        prediction.square().mean().backward()
        optimizer.step()
    assert prediction.shape == (2, 12)
    assert aux["spatial_attention"].shape == (2, 20, 1, 64)
    assert torch.allclose(
        aux["spatial_attention"].sum(dim=-1),
        torch.ones(2, 20, 1, device=device),
        atol=1e-5,
    )
    assert torch.equal(components["wind_base"], components["ar_base"])
    assert torch.count_nonzero(model.column_projection[0].weight.grad) > 0
    assert torch.isfinite(prediction).all()

    model.eval()
    dropped, dropped_components = model(
        images,
        wind,
        return_components=True,
        time_keep=torch.ones(2, 20, device=device),
        image_keep=torch.zeros(2, device=device),
    )
    assert torch.equal(dropped, dropped_components["ar_base"])
    assert torch.count_nonzero(dropped_components["image_correction"]) == 0
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        "v16 native profile LSTM smoke passed: "
        f"device={device} parameters={parameter_count:,} "
        f"prediction={tuple(prediction.shape)} columns=64"
    )


if __name__ == "__main__":
    main()
