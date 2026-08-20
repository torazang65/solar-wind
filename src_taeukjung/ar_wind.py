from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlobalARFit:
    order: int
    ridge_strength: float
    coefficients: np.ndarray
    intercept: float
    transition_count: int
    series_lengths: tuple


def reconstruct_wind_series(inputs, targets, temporal_chains, wind_columns):
    """Recover each unique train timeline and append its supervised tail."""
    wind = inputs[wind_columns].to_numpy(np.float64) / 1000.0
    target = np.asarray(targets, dtype=np.float64) / 1000.0
    series = []

    for chain in temporal_chains.chains:
        values = wind[chain[0]].tolist()
        for previous_row, row_index in zip(chain[:-1], chain[1:]):
            if not np.array_equal(wind[previous_row, 1:], wind[row_index, :-1]):
                raise ValueError("image and wind sliding windows are inconsistent")
            values.append(float(wind[row_index, -1]))

        final_row = chain[-1]
        values.extend(target[final_row].tolist())
        timeline = np.asarray(values, dtype=np.float64)

        for position, row_index in enumerate(chain):
            expected = timeline[position + 20 : position + 32]
            if not np.array_equal(target[row_index], expected):
                raise ValueError("reconstructed train wind does not match targets")
        series.append(timeline)
    return tuple(series)


def fit_global_ar(
    inputs,
    targets,
    temporal_chains,
    wind_columns,
    order=2,
    ridge_strength=30.0,
):
    """Fit one regularized AR(p) process to all unique train transitions."""
    if order <= 0 or order > len(wind_columns):
        raise ValueError("AR order must be between 1 and the observed wind length")
    if ridge_strength < 0.0:
        raise ValueError("ridge_strength must be nonnegative")

    series = reconstruct_wind_series(
        inputs, targets, temporal_chains, wind_columns
    )
    features = []
    responses = []
    for timeline in series:
        for index in range(order, len(timeline)):
            features.append(timeline[index - order : index])
            responses.append(timeline[index])
    features = np.asarray(features, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    response_mean = float(responses.mean())
    standardized = (features - feature_mean) / feature_scale
    system = standardized.T @ standardized
    system += ridge_strength * np.eye(order, dtype=np.float64)
    standardized_coefficients = np.linalg.solve(
        system, standardized.T @ (responses - response_mean)
    )
    coefficients = standardized_coefficients / feature_scale
    intercept = response_mean - float(feature_mean @ coefficients)
    return GlobalARFit(
        order=int(order),
        ridge_strength=float(ridge_strength),
        coefficients=coefficients.astype(np.float32),
        intercept=float(intercept),
        transition_count=int(len(responses)),
        series_lengths=tuple(len(timeline) for timeline in series),
    )


def predict_recursive_ar(wind, coefficients, intercept, forecast_steps=12):
    wind = np.asarray(wind, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if wind.ndim != 2 or wind.shape[1] < len(coefficients):
        raise ValueError("wind history is shorter than the AR order")
    history = [wind[:, index].copy() for index in range(wind.shape[1])]
    predictions = []
    for _ in range(forecast_steps):
        context = np.stack(history[-len(coefficients) :], axis=1)
        next_value = float(intercept) + context @ coefficients
        history.append(next_value)
        predictions.append(next_value)
    return np.stack(predictions, axis=1)


def residual_scale(targets, predictions):
    residual = np.asarray(targets, dtype=np.float64) / 1000.0 - predictions
    centered = residual - residual.mean(axis=0, keepdims=True)
    return np.sqrt(np.maximum(np.mean(centered**2, axis=0), 1e-8)).astype(
        np.float32
    )


def validation_metrics(targets, predictions, chain_ids):
    error_km_s = (
        np.asarray(predictions, dtype=np.float64)
        - np.asarray(targets, dtype=np.float64) / 1000.0
    ) * 1000.0
    micro = float(np.sqrt(np.mean(error_km_s**2)))
    macro = float(
        np.mean(
            [
                np.sqrt(np.mean(error_km_s[chain_ids == chain_id] ** 2))
                for chain_id in np.unique(chain_ids)
            ]
        )
    )
    return micro, macro
