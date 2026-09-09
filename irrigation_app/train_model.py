"""
train_model.py  ——  standalone script that reproduces the notebook's training
pipeline and saves model/irrigation_model.joblib.

Run this ONCE from the irrigation_app/ directory:
    python train_model.py

Mirrors Sections 2–8 of AI_Irrigation_Optimizer_v4_larger_real_data.ipynb.

── Real data files (optional, place in data/) ──────────────────────────────
  data/Crop_recommendation.csv   Kaggle: atharvaingle/crop-recommendation-dataset
  data/mendeley_soil_motor.csv   Mendeley soil/motor sensor logs
  data/zindi_train.csv           Zindi WaziHub soil-moisture challenge Train.csv

Each source is loaded if present; missing files are skipped silently and the
corresponding columns are filled from synthetic distributions — exactly as the
notebook does via its pool() function.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Reference tables (identical to notebook) ──────────────────────────────────
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
DEFAULT_KC, DEFAULT_ROOT_DEPTH = 0.85, 900

crops  = list(KC_TABLE.keys())
soils  = ['Sandy', 'Loamy', 'Clayey']
stages = ['Initial', 'Development', 'Mid-season', 'Late-season']

# ── Helper functions ──────────────────────────────────────────────────────────
CROP_SYNONYMS = {
    'paddy': 'Rice', 'rice': 'Rice', 'wheat': 'Wheat', 'cotton': 'Cotton',
    'sugarcane': 'Sugarcane', 'maize': 'Maize', 'corn': 'Maize',
    'coffee': 'Coffee', 'chickpea': 'Chickpea', 'banana': 'Banana',
}

def normalize_crop(name):
    return CROP_SYNONYMS.get(str(name).strip().lower(), name if name in KC_TABLE else 'Maize')

def estimate_et0(temp, humidity, solar=18.0):
    return np.clip(
        0.0023 * (solar * 30) * (np.asarray(temp) + 17.8)
        * (1 - np.asarray(humidity) / 100) ** 0.5,
        1, 12,
    )

def synth_target(temp, humidity, rainfall, soil_moist, field_cap, crop, stage):
    kc = np.array([KC_TABLE.get(c, {}).get(s, DEFAULT_KC) for c, s in zip(crop, stage)])
    et0 = estimate_et0(temp, humidity)
    deficit = np.clip(np.asarray(field_cap) - np.asarray(soil_moist), 0, None)
    return np.clip(kc * et0 + 0.1 * deficit - np.asarray(rainfall) * 0.75, 0, None)

# ── Real data loaders (mirror notebook Section 3) ─────────────────────────────
def load_crop_recommendation():
    """Kaggle: atharvaingle/crop-recommendation-dataset"""
    path = os.path.join('data', 'Crop_recommendation.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out['temperature_c']        = df['temperature']
    out['humidity_pct']         = df['humidity']
    out['rainfall_forecast_mm'] = df['rainfall'] / 30.0  # annual->daily scale
    out['crop_type']            = df['label'].apply(normalize_crop)
    print(f"  Crop Recommendation CSV: {len(out)} real rows loaded")
    return out

def load_mendeley_motor():
    """Mendeley soil/motor sensor logs — actual columns: Soil Moisture, Temperature, Air Humidity"""
    matches = (glob.glob(os.path.join('data', 'mendeley_soil_motor*.csv')) or
               glob.glob(os.path.join('data', '*motor*.csv')))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    # Real column names (case-sensitive preserved, so we map explicitly)
    col_map = {c.strip(): c for c in df.columns}
    out = pd.DataFrame()
    # 'Soil Moisture' is a raw sensor reading (0–1000); normalise to 0–100 %
    sm_col = col_map.get('Soil Moisture')
    if sm_col:
        raw = pd.to_numeric(df[sm_col], errors='coerce')
        out['soil_moisture_pct'] = (raw / 10.0).clip(0, 100)   # 0-1000 -> 0-100%
    # Temperature
    temp_col = col_map.get('Temperature')
    if temp_col:
        out['temperature_c'] = pd.to_numeric(df[temp_col], errors='coerce')
    # Air Humidity
    hum_col = col_map.get('Air Humidity')
    if hum_col:
        out['humidity_pct'] = pd.to_numeric(df[hum_col], errors='coerce')
    if 'soil_moisture_pct' not in out.columns:
        return None
    out = out.dropna(subset=['soil_moisture_pct'])
    print(f"  Mendeley CSV: {len(out)} real rows loaded  (soil range: {out['soil_moisture_pct'].min():.1f}–{out['soil_moisture_pct'].max():.1f}%)")
    return out

def load_zindi_wazihub():
    """Zindi WaziHub — actual columns: Soil humidity 1..4, Air temperature (C), Air humidity (%), Wind speed (Km/h)"""
    path = os.path.join('data', 'zindi_train.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = pd.DataFrame()
    # Average the four soil humidity sensors (all in %)
    soil_cols = [c for c in df.columns if c.startswith('Soil humidity')]
    if soil_cols:
        out['soil_moisture_pct'] = df[soil_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    if 'Air temperature (C)' in df.columns:
        out['temperature_c'] = pd.to_numeric(df['Air temperature (C)'], errors='coerce')
    if 'Air humidity (%)' in df.columns:
        out['humidity_pct'] = pd.to_numeric(df['Air humidity (%)'], errors='coerce')
    if 'Wind speed (Km/h)' in df.columns:
        out['wind_speed_kmph'] = pd.to_numeric(df['Wind speed (Km/h)'], errors='coerce')
    if 'soil_moisture_pct' not in out.columns:
        return None
    out = out[out['soil_moisture_pct'] >= 0]   # drop -1.xx bad sensor readings
    out = out.dropna(subset=['soil_moisture_pct'])
    print(f"  Zindi WaziHub CSV: {len(out)} real rows loaded  (soil range: {out['soil_moisture_pct'].min():.1f}–{out['soil_moisture_pct'].max():.1f}%)")
    return out

# ── pool() helper (exact copy from notebook Section 4) ────────────────────────
def pool(series, n, fallback):
    if series is not None and len(series.dropna()) > 0:
        return series.dropna().sample(n, replace=True).reset_index(drop=True)
    return pd.Series(
        np.random.choice(fallback, n) if isinstance(fallback, list)
        else np.random.uniform(*fallback, n)
    )

# ── Load available real sources ────────────────────────────────────────────────
print("Loading real data sources from data/ …")
real_crop = load_crop_recommendation()
real_soil_sources = [d for d in [load_mendeley_motor(), load_zindi_wazihub()] if d is not None]
real_soil = pd.concat(real_soil_sources, ignore_index=True) if real_soil_sources else None

if real_crop is None and real_soil is None:
    print("  No real data files found — using fully synthetic distributions.")
print()

# ── Build training set (8 000 rows, notebook Section 4 pool() logic) ──────────
N = 8000
print(f"Building training set ({N} rows) …")

df = pd.DataFrame()
df['temperature_c']        = pool(
    real_crop['temperature_c'] if real_crop is not None else None, N, (10, 48))
df['humidity_pct']         = pool(
    real_crop['humidity_pct'] if real_crop is not None else None, N, (10, 95))
df['wind_speed_kmph']      = pool(
    real_soil['wind_speed_kmph'] if real_soil is not None and 'wind_speed_kmph' in real_soil.columns else None,
    N, (0, 40))
df['rainfall_forecast_mm'] = pool(
    real_crop['rainfall_forecast_mm'] if real_crop is not None else None, N, (0, 26))
df['soil_moisture_pct']    = pool(
    real_soil['soil_moisture_pct'] if real_soil is not None else None, N, (5, 45))
df['field_capacity_pct']   = (df['soil_moisture_pct'] + np.random.uniform(5, 20, N)).clip(upper=45)
df['crop_type']            = pool(
    real_crop['crop_type'] if real_crop is not None else None, N, crops)
df['soil_type']            = np.random.choice(soils, N)
df['growth_stage']         = np.random.choice(stages, N)
df['et0_mm']               = estimate_et0(df['temperature_c'], df['humidity_pct'])
df['irrigation_required_mm'] = synth_target(
    df['temperature_c'], df['humidity_pct'], df['rainfall_forecast_mm'],
    df['soil_moisture_pct'], df['field_capacity_pct'],
    df['crop_type'], df['growth_stage'],
)

# ── Feature engineering (identical column order to notebook Section 5) ────────
df['moisture_deficit_pct'] = (df['field_capacity_pct'] - df['soil_moisture_pct']).clip(lower=0)
df['rain_adjusted_et0']    = (df['et0_mm'] - df['rainfall_forecast_mm'] * 0.1).clip(lower=0)
df['kc']                   = df.apply(
    lambda r: KC_TABLE.get(r['crop_type'], {}).get(r['growth_stage'], DEFAULT_KC), axis=1
)
df['root_zone_depth_mm']   = df['crop_type'].map(ROOT_ZONE_DEPTH_MM).fillna(DEFAULT_ROOT_DEPTH)

feature_cols_num = [
    'temperature_c', 'humidity_pct', 'wind_speed_kmph', 'rainfall_forecast_mm',
    'soil_moisture_pct', 'field_capacity_pct', 'et0_mm', 'kc', 'root_zone_depth_mm',
    'moisture_deficit_pct', 'rain_adjusted_et0',
]
feature_cols_cat = ['crop_type', 'soil_type', 'growth_stage']

X = df[feature_cols_num + feature_cols_cat]
y = df['irrigation_required_mm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train/test split: {X_train.shape} / {X_test.shape}")

# ── Preprocessor ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', StandardScaler(),             feature_cols_num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols_cat),
])

# ── Train & select best model ─────────────────────────────────────────────────
models = {
    'Linear Regression':   LinearRegression(),
    'Random Forest':       RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingRegressor(random_state=42),
}

results, pipelines = {}, {}
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse  = mean_squared_error(y_test, preds) ** 0.5
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    results[name]  = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    pipelines[name] = pipe
    print(f"  {name:25s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

results_df = pd.DataFrame(results).T.sort_values('RMSE')
best_name  = results_df.index[0]
best_pipe  = pipelines[best_name]
print(f"\nBest model: {best_name}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
out_path = os.path.join('model', 'irrigation_model.joblib')
joblib.dump(best_pipe, out_path)
print(f"Saved -> {out_path}")

# ── Quick smoke-test with Section 9 demo values ───────────────────────────────
print("\n── Section 9 demo verification ──────────────────────────────────")
sample = {
    'temperature_c': 42, 'humidity_pct': 30, 'wind_speed_kmph': 4,
    'rainfall_forecast_mm': 0, 'soil_moisture_pct': 20, 'field_capacity_pct': 33,
    'et0_mm': 8.6, 'crop_type': 'Rice', 'soil_type': 'Loamy',
    'growth_stage': 'Mid-season',
}
from logic import recommend_irrigation
result = recommend_irrigation(best_pipe, sample, field_area_m2=2000, flow_rate_lpm=200)
for k, v in result.items():
    print(f"  {k:30s}: {v}")
