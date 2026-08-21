import torch

from model_solar_timing_transformer_v13 import (
    FORECAST_STEPS,
    HINDCAST_STEPS,
    QUERY_STEPS,
    SolarWindTimingTransformerV13,
)


def main():
    torch.manual_seed(777)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SolarWindTimingTransformerV13(
        image_size=64,
        ar_coefficients=[-0.34, 1.28],
        ar_intercept=0.025,
        baseline_residual_scale=[0.08] * FORECAST_STEPS,
        grid_rows=2,
        grid_columns=8,
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
    assert aux["hindcast"].shape == (1, HINDCAST_STEPS)
    assert aux["timing_attention"].shape == (1, QUERY_STEPS, 20 * 2 * 8)
    assert aux["source_speed"].shape == (1, 20, 2, 8)
    assert torch.allclose(
        aux["timing_attention"].sum(dim=-1),
        torch.ones(1, QUERY_STEPS, device=device),
        atol=1e-5,
    )
    earliest_attention = aux["timing_attention"][0, 0].view(20, 2, 8)
    assert torch.count_nonzero(earliest_attention[8:] > 1e-6) == 0
    assert aux["source_speed"].std() > 1e-4
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(model.source_speed_head.weight.grad) > 0
    assert torch.count_nonzero(
        model.query_blocks[0].key_projection.weight.grad
    ) > 0
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
        "v13 timing Transformer smoke passed: "
        f"device={device} "
        f"prediction={tuple(prediction.shape)} "
        f"hindcast={tuple(aux['hindcast'].shape)} "
        f"attention={tuple(aux['timing_attention'].shape)}"
    )


if __name__ == "__main__":
    main()
