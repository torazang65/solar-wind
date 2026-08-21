import torch
from torch.nn import functional as F

from model_solar_peak_event_v15 import SolarWindPeakEventV15


def main():
    torch.manual_seed(777)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SolarWindPeakEventV15(
        image_size=64,
        ar_coefficients=[-0.34, 1.28],
        ar_intercept=0.025,
        baseline_residual_scale=[0.08] * 12,
        grid_rows=2,
        grid_columns=8,
        dropout=0.0,
        time_mask_prob=0.0,
        modality_drop_prob=0.0,
        deformable_points=8,
    ).to(device)
    model.train()
    images = torch.rand(2, 20, 2, 64, 64, device=device)
    wind = 0.35 + 0.15 * torch.rand(2, 20, device=device)
    target = 0.35 + 0.25 * torch.rand(2, 12, device=device)
    prediction, components, aux = model(
        images, wind, return_components=True, return_aux=True
    )
    assert prediction.shape == (2, 12)
    assert aux["peak_time_logits"].shape == (2, 12)
    assert aux["peak_time_probability"].shape == (2, 12)
    assert aux["peak_value"].shape == (2,)
    assert torch.allclose(
        aux["peak_time_probability"].sum(dim=-1),
        torch.ones(2, device=device),
        atol=1e-6,
    )
    target_peak_value, target_peak_index = target.max(dim=-1)
    time_loss = F.cross_entropy(aux["peak_time_logits"], target_peak_index)
    value_loss = torch.sqrt(
        F.mse_loss(aux["peak_value"], target_peak_value) + 1e-8
    )
    (F.mse_loss(prediction, target) + 0.05 * time_loss + 0.25 * value_loss).backward()
    assert torch.count_nonzero(model.peak_time_head.weight.grad) > 0
    assert torch.count_nonzero(model.peak_value_head[-1].weight.grad) > 0
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
    print(
        "v15 direct peak-event smoke passed: "
        f"device={device} prediction={tuple(prediction.shape)} "
        f"peak_time={tuple(aux['peak_time_logits'].shape)} "
        f"peak_value={tuple(aux['peak_value'].shape)}"
    )


if __name__ == "__main__":
    main()
