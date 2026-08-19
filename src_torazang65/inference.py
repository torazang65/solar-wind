import gc
import numpy as np
import pandas as pd
import torch
from config import *
from model import SolarWindBaseline
from dataset import (
    val_loader, val_targets, val_index, val_inputs, 
    test_image_array, test_image_index, test_inputs, test_index,
    SolarWindDataset, make_loader, TARGET_COLUMNS, WIND_COLUMNS
)

def load_best_model():
    model = SolarWindBaseline(image_size=IMAGE_SIZE).to(DEVICE)
    checkpoint_path = OUTPUT_DIR / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("loaded best epoch:", checkpoint["epoch"])
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions, sample_ids = [], []
    for batch in loader:
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
            output = model(images, wind)
        predictions.append(output.float().cpu().numpy() * 1000.0)
        sample_ids.extend(batch["sample_id"])
    return np.concatenate(predictions), sample_ids

def metrics_by_horizon(y_true, y_pred):
    rows = []
    for index in range(12):
        actual = y_true[:, index]
        predicted = y_pred[:, index]
        error = predicted - actual
        denominator = np.std(actual) * np.std(predicted)
        correlation = float(np.corrcoef(actual, predicted)[0, 1]) if denominator > 0 else np.nan
        rows.append({
            "horizon_hours": (index + 1) * 6,
            "rmse_km_s": float(np.sqrt(np.mean(error ** 2))),
            "mae_km_s": float(np.mean(np.abs(error))),
            "correlation": correlation,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    model = load_best_model()

    # --- Validation Evaluation ---
    print("\n[Running Validation Evaluation]")
    validation_prediction, validation_ids = predict(model, val_loader)
    validation_target = np.asarray(val_targets[val_index], dtype=np.float64)
    
    validation_persistence = np.repeat(
        val_inputs.iloc[val_index][[WIND_COLUMNS[-1]]].to_numpy(np.float64), 12, axis=1
    )
    
    validation_metrics = metrics_by_horizon(validation_target, validation_prediction)
    persistence_metrics = metrics_by_horizon(validation_target, validation_persistence)
    validation_metrics["persistence_rmse_km_s"] = persistence_metrics.rmse_km_s
    validation_metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    
    print("Overall validation RMSE:", float(np.sqrt(np.mean((validation_prediction - validation_target) ** 2))))
    
    # --- Clean up Memory before Test ---
    del val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # --- Test Inference ---
    print("\n[Running Test Inference]")
    test_dataset = SolarWindDataset(test_image_array, test_image_index, test_inputs, test_index, targets=None)
    test_loader = make_loader(test_dataset, shuffle=False)
    
    test_prediction, predicted_ids = predict(model, test_loader)
    
    submission = pd.DataFrame(test_prediction, columns=TARGET_COLUMNS)
    submission.insert(0, "sample_id", predicted_ids)
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print("saved:", submission_path.resolve())
    print("shape:", submission.shape)