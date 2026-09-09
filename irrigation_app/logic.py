"""
logic.py — AI Irrigation Optimizer
Ported faithfully from AI_Irrigation_Optimizer_v4_larger_real_data.ipynb.
DO NOT retrain the model here; load the pre-trained .joblib artifact.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# 1.  Reference tables (exact copies from notebook Section 2)
# ---------------------------------------------------------------------------

KC_TABLE = {
    'Rice':      {'Initial': 1.05, 'Development': 1.10, 'Mid-season': 1.20, 'Late-season': 0.90},
    'Wheat':     {'Initial': 0.40, 'Development': 0.75, 'Mid-season': 1.15, 'Late-season': 0.40},
    'Cotton':    {'Initial': 0.35, 'Development': 0.70, 'Mid-season': 1.15, 'Late-season': 0.65},
    'Sugarcane': {'Initial': 0.40, 'Development': 0.85, 'Mid-season': 1.25, 'Late-season': 0.75},
    'Maize':     {'Initial': 0.30, 'Development': 0.70, 'Mid-season': 1.20, 'Late-season': 0.60},
    'Coffee':    {'Initial': 0.90, 'Development': 0.95, 'Mid-season': 1.05, 'Late-season': 0.95},
    'Chickpea':  {'Initial': 0.40, 'Development': 0.70, 'Mid-season': 1.00, 'Late-season': 0.35},
    'Banana':    {'Initial': 0.50, 'Development': 0.90, 'Mid-season': 1.10, 'Late-season': 1.00},
}

ROOT_ZONE_DEPTH_MM = {
    'Rice': 400, 'Wheat': 900, 'Cotton': 1200, 'Sugarcane': 1500,
    'Maize': 1000, 'Coffee': 1200, 'Chickpea': 700, 'Banana': 900,
}

CROP_SYNONYMS = {
    'paddy': 'Rice', 'rice': 'Rice', 'wheat': 'Wheat', 'cotton': 'Cotton',
    'sugarcane': 'Sugarcane', 'maize': 'Maize', 'corn': 'Maize',
    'coffee': 'Coffee', 'chickpea': 'Chickpea', 'banana': 'Banana',
}

SOIL_SYNONYMS = {
    'alluvial': 'Loamy', 'sandy': 'Sandy', 'loamy': 'Loamy',
    'clayey': 'Clayey', 'clay': 'Clayey', 'silty': 'Loamy', 'peaty': 'Loamy',
}

DEFAULT_KC = 0.85
DEFAULT_ROOT_DEPTH = 900

# ---------------------------------------------------------------------------
# 2.  Feature schema (exact order from notebook Section 5)
# ---------------------------------------------------------------------------

feature_cols_num = [
    'temperature_c', 'humidity_pct', 'wind_speed_kmph', 'rainfall_forecast_mm',
    'soil_moisture_pct', 'field_capacity_pct', 'et0_mm', 'kc', 'root_zone_depth_mm',
    'moisture_deficit_pct', 'rain_adjusted_et0',
]
feature_cols_cat = ['crop_type', 'soil_type', 'growth_stage']

# ---------------------------------------------------------------------------
# 3.  Helper functions (exact copies from notebook)
# ---------------------------------------------------------------------------

def normalize_crop(name: str) -> str:
    return CROP_SYNONYMS.get(str(name).strip().lower(), name if name in KC_TABLE else 'Maize')


def normalize_soil(name: str) -> str:
    return SOIL_SYNONYMS.get(
        str(name).strip().lower(),
        name if name in ['Sandy', 'Loamy', 'Clayey'] else 'Loamy',
    )


def estimate_et0(temp, humidity, solar: float = 18.0) -> float:
    """Simplified Hargreaves-style ET0 estimate (mm/day)."""
    return float(
        np.clip(
            0.0023 * (solar * 30) * (np.asarray(temp) + 17.8)
            * (1 - np.asarray(humidity) / 100) ** 0.5,
            1,
            12,
        )
    )


def get_confidence(pipeline, input_df: pd.DataFrame) -> float:
    """Return confidence % based on tree ensemble spread, else default 75."""
    model = pipeline.named_steps['model']
    if hasattr(model, 'estimators_'):
        X_t = pipeline.named_steps['preprocessor'].transform(input_df)
        trees = np.array(model.estimators_).ravel()
        tree_preds = np.array([t.predict(X_t)[0] for t in trees])
        if tree_preds.mean() > 0:
            spread = tree_preds.std() / (abs(tree_preds.mean()) + 1e-6)
            return float(np.clip(100 - spread * 100, 40, 99))
    return 75.0


def recommend_irrigation(
    pipeline,
    reading: dict,
    field_area_m2: float = 1000.0,
    flow_rate_lpm: float = 200.0,
) -> dict:
    """
    Decision engine — exact copy from notebook Section 7.
    Returns dict with keys: irrigation_required, need_level, reason,
    raw_model_prediction_mm, recommended_water_liters,
    recommended_time_minutes, confidence_pct.
    """
    r = dict(reading)
    r['crop_type'] = normalize_crop(r['crop_type'])
    r['soil_type']  = normalize_soil(r['soil_type'])
    r['kc']                   = KC_TABLE.get(r['crop_type'], {}).get(r['growth_stage'], DEFAULT_KC)
    r['root_zone_depth_mm']   = ROOT_ZONE_DEPTH_MM.get(r['crop_type'], DEFAULT_ROOT_DEPTH)
    r['moisture_deficit_pct'] = max(0, r['field_capacity_pct'] - r['soil_moisture_pct'])
    r['rain_adjusted_et0']    = max(0, r['et0_mm'] - r['rainfall_forecast_mm'] * 0.1)

    input_df = pd.DataFrame([r])[feature_cols_num + feature_cols_cat]
    predicted_mm = float(pipeline.predict(input_df)[0])
    confidence   = get_confidence(pipeline, input_df)

    # --- Decision tiers (exact thresholds from notebook) ---
    if r['soil_moisture_pct'] >= r['field_capacity_pct']:
        tier, level, reason = "NO", "LOW", "Soil already at/above field capacity — waterlogging risk."
        final_mm = 0.0
    elif r['rainfall_forecast_mm'] > 10:
        tier, level, reason = (
            "NO", "LOW",
            f"Significant rain forecast ({r['rainfall_forecast_mm']} mm) — wait and re-check.",
        )
        final_mm = 0.0
    elif predicted_mm < 1.0:
        tier, level, reason = "NO", "LOW", "Predicted need is negligible (<1 mm)."
        final_mm = 0.0
    elif predicted_mm < 4.0:
        tier, level, reason = "YES", "MODERATE", "Moderate water deficit for current crop stage/conditions."
        final_mm = round(predicted_mm, 2)
    else:
        tier, level, reason = "YES", "HIGH", "High water deficit — irrigate soon to avoid crop stress."
        final_mm = round(predicted_mm, 2)

    liters  = round(final_mm * field_area_m2, 1)
    minutes = round(liters / flow_rate_lpm, 1) if liters > 0 else 0.0

    return {
        "irrigation_required":      tier,
        "need_level":               level,
        "reason":                   reason,
        "raw_model_prediction_mm":  round(predicted_mm, 2),
        "recommended_water_liters": liters,
        "recommended_time_minutes": minutes,
        "confidence_pct":           round(confidence, 1),
    }

# ---------------------------------------------------------------------------
# 4.  Load trained pipeline at import time
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "irrigation_model.joblib")


def _load_pipeline():
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"\n\n[ERROR] Model file not found: {_MODEL_PATH}\n"
            "Please run the Jupyter notebook "
            "(AI_Irrigation_Optimizer_v4_larger_real_data.ipynb) to train and save "
            "the model, then copy 'artifacts/irrigation_model.joblib' to "
            "'irrigation_app/model/irrigation_model.joblib' before starting the app.\n"
        )
    return joblib.load(_MODEL_PATH)


try:
    pipeline = _load_pipeline()
    MODEL_LOADED = True
    MODEL_ERROR  = None
except FileNotFoundError as exc:
    pipeline     = None
    MODEL_LOADED = False
    MODEL_ERROR  = str(exc)
except Exception as exc:
    pipeline     = None
    MODEL_LOADED = False
    MODEL_ERROR  = (
        f"[ERROR] Failed to load model: {exc}\n"
        "This often means a scikit-learn version mismatch. "
        "Ensure the same version used during training is installed."
    )
