import torch
from torch.nn import functional as F

from model_solar_transport_fusion_v17 import SolarWindTransportFusionV17


def trainable_gradient_count(module):
    return sum(
        int(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main():
    torch.manual_seed(777)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = SolarWindTransportFusionV17(
        image_size=64,
        ar_coefficients=[-0.34, 1.28],
        ar_intercept=0.025,
        baseline_residual_scale=[0.08] * 12,
        dropout=0.0,
        time_mask_prob=0.0,
        modality_drop_prob=0.0,
    ).to(device)
    images = torch.rand(2, 20, 2, 64, 64, device=device)
    wind = 0.35 + 0.15 * torch.rand(2, 20, device=device)
    target = 0.35 + 0.15 * torch.rand(2, 12, device=device)

    delay = model.physical_delay_hours()
    assert delay.shape == (64, 5)
    assert torch.all(delay >= model.minimum_delay_hours)
    assert torch.all(delay <= model.maximum_delay_hours)
    center = model.image_size // 2
    assert torch.all(delay[center, 1:] < delay[center, :-1])

    model.set_stage("transport")
    prediction, components, aux = model(
        images, wind, return_components=True, return_aux=True
    )
    assert prediction.shape == (2, 12)
    assert aux["transport_hindcast"].shape == (2, 10)
    assert aux["transport_forecast"].shape == (2, 12)
    assert aux["expert_probability"].shape == (2, 20, 64, 5)
    assert torch.allclose(
        aux["expert_probability"].sum(dim=-1),
        torch.ones(2, 20, 64, device=device),
        atol=1e-5,
    )
    transport_loss = F.mse_loss(
        aux["transport_hindcast"], aux["hindcast_wind"]
    )
    transport_loss.backward()
    assert trainable_gradient_count(model.profile_projection) > 0
    assert all(parameter.grad is None for parameter in model.fusion_head.parameters())

    model.zero_grad(set_to_none=True)
    model.set_stage("fusion")
    prediction = model(images, wind)
    F.mse_loss(prediction, target).backward()
    assert trainable_gradient_count(model.fusion_head) > 0
    assert all(
        parameter.grad is None for parameter in model.profile_projection.parameters()
    )

    model.eval()
    full, full_components, full_aux = model(
        images,
        wind,
        return_components=True,
        return_aux=True,
        time_keep=torch.ones(2, 20, device=device),
        image_keep=torch.ones(2, device=device),
    )
    future_changed = images.clone()
    future_changed[:, 10:] = 1.0 - future_changed[:, 10:]
    _, changed_aux = model(
        future_changed,
        wind,
        return_aux=True,
        time_keep=torch.ones(2, 20, device=device),
        image_keep=torch.ones(2, device=device),
    )
    assert torch.allclose(
        full_aux["transport_hindcast"][:, 0],
        changed_aux["transport_hindcast"][:, 0],
        atol=1e-6,
    )
    dropped, dropped_components = model(
        images,
        wind,
        return_components=True,
        time_keep=torch.ones(2, 20, device=device),
        image_keep=torch.zeros(2, device=device),
    )
    assert torch.equal(dropped, dropped_components["ar_base"])
    assert torch.count_nonzero(dropped_components["image_correction"]) == 0
    assert not torch.equal(full, full_components["ar_base"])
    assert torch.isfinite(full).all()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "v17 transport fusion smoke passed: "
        f"device={device} parameters={parameter_count:,} "
        f"prediction={tuple(full.shape)} native_columns=64 experts=5"
    )


if __name__ == "__main__":
    main()
