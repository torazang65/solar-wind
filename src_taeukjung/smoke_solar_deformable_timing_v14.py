import torch

from model_solar_deformable_timing_v14 import (
    FORECAST_STEPS,
    HINDCAST_STEPS,
    QUERY_STEPS,
    SolarWindDeformableTimingV14,
)


def main():
    torch.manual_seed(777)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SolarWindDeformableTimingV14(
        image_size=64,
        ar_coefficients=[-0.34, 1.28],
        ar_intercept=0.025,
        baseline_residual_scale=[0.08] * FORECAST_STEPS,
        grid_rows=2,
        grid_columns=8,
        dropout=0.0,
        time_mask_prob=0.0,
        modality_drop_prob=0.0,
        deformable_points=8,
    ).to(device)
    model.train()
    images = torch.rand(1, 20, 2, 64, 64, device=device)
    wind = 0.35 + 0.15 * torch.rand(1, 20, device=device)
    prediction, components, aux = model(
        images, wind, return_components=True, return_aux=True
    )
    assert prediction.shape == (1, FORECAST_STEPS)
    assert aux["hindcast"].shape == (1, HINDCAST_STEPS)
    assert aux["query_features"].shape == (1, QUERY_STEPS, model.d_model)
    assert aux["sparse_attention"].shape == (
        1,
        model.attention_heads,
        QUERY_STEPS,
        model.deformable_points,
    )
    assert aux["timing_attention"].shape == (
        1,
        QUERY_STEPS,
        20 * model.grid_rows * model.grid_columns,
    )
    assert torch.allclose(
        aux["sparse_attention"].sum(dim=-1),
        torch.ones_like(aux["sparse_attention"].sum(dim=-1)),
        atol=1e-5,
    )
    assert torch.allclose(
        aux["timing_attention"].sum(dim=-1),
        torch.ones_like(aux["timing_attention"].sum(dim=-1)),
        atol=1e-5,
    )
    earliest_attention = aux["timing_attention"][0, 0].view(20, 2, 8)
    assert torch.count_nonzero(earliest_attention[8:] > 1e-7) == 0
    assert aux["source_speed"].std() > 1e-4
    assert torch.isfinite(prediction).all()
    (prediction.square().mean() + aux["hindcast"].square().mean()).backward()
    offset_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "offset_projection" in name and parameter.grad is not None
    ]
    assert offset_gradients
    assert sum(gradient.abs().sum() for gradient in offset_gradients) > 0
    assert model.effective_distance_raw.grad.abs() > 0

    model.eval()
    dropped, dropped_components = model(
        images,
        wind,
        return_components=True,
        time_keep=torch.ones(1, 20, device=device),
        image_keep=torch.zeros(1, device=device),
    )
    assert torch.equal(dropped, dropped_components["ar_base"])
    assert torch.count_nonzero(dropped_components["image_correction"]) == 0
    print(
        "v14 deformable timing smoke passed: "
        f"device={device} prediction={tuple(prediction.shape)} "
        f"sparse={tuple(aux['sparse_attention'].shape)} "
        f"dense={tuple(aux['timing_attention'].shape)}"
    )


if __name__ == "__main__":
    main()
