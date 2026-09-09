# 🌱 AI Irrigation Optimizer — Flask Web App

A locally-runnable Flask web application that wraps a trained scikit-learn irrigation-recommendation pipeline.

## Prerequisites

- Python 3.9+
- The trained model file (see **Step 0** below)

---

## Step 0 — Export the trained model from the notebook

> **You must do this once before running the app.**

1. Open `AI_Irrigation_Optimizer_v4_larger_real_data.ipynb` in Jupyter / Google Colab.
2. Run **all cells** (Runtime → Run all).
3. After Section 8 ("Save Model") runs, you will have `artifacts/irrigation_model.joblib`.
4. Copy that file into the app's model folder:
   ```
   irrigation_app/model/irrigation_model.joblib
   ```
   If running in Colab, use `files.download('artifacts/irrigation_model.joblib')` (uncomment the line in Section 8) and then move it to the folder above.

---

## Step 1 — Install dependencies

```bash
cd irrigation_app
pip install -r requirements.txt
```

---

## Step 2 — Run the app

```bash
flask --app app run
```

Or equivalently:
```bash
python app.py
```

The server starts at **http://127.0.0.1:5000/**

---

## Step 3 — Verify against the notebook's demo (Section 9)

Use these values in the web form — they match the notebook's `sample_reading`:

| Field                | Value            |
|----------------------|------------------|
| Temperature          | 42 °C            |
| Relative Humidity    | 30 %             |
| Wind Speed           | 4 km/h           |
| Rainfall Forecast    | 0 mm             |
| Soil Moisture        | 20 %             |
| Field Capacity       | 33 %             |
| Crop Type            | Rice (= Paddy)   |
| Soil Type            | Loamy (= Alluvial) |
| Growth Stage         | Mid-season       |
| Field Area           | 2000 m²          |
| Flow Rate            | 200 L/min        |

Expected result: `irrigation_required = YES`, `need_level = HIGH`, with `raw_model_prediction_mm ≈ 7–10 mm` (exact value depends on which model was selected as best by the notebook).

---

## JSON API

The `/predict` endpoint also accepts requests with `Accept: application/json`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Accept: application/json" \
  -d "temperature_c=42&humidity_pct=30&wind_speed_kmph=4&rainfall_forecast_mm=0&soil_moisture_pct=20&field_capacity_pct=33&crop_type=Rice&soil_type=Loamy&growth_stage=Mid-season&field_area_m2=2000&flow_rate_lpm=200"
```

---

## Project structure

```
irrigation_app/
├── app.py              Flask application (routes)
├── logic.py            Model logic ported from notebook (no retraining)
├── model/
│   └── irrigation_model.joblib   ← copy from notebook artifacts/
├── templates/
│   └── index.html      Single-page HTML form + result display
├── static/
│   └── style.css       Dark-mode glassmorphism styling
├── requirements.txt
└── README.md
```

---

## Known risks / troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: model/irrigation_model.joblib` | Complete Step 0 above |
| scikit-learn version mismatch error on load | Use the same scikit-learn version as the notebook environment |
| Prediction seems wrong | Ensure the notebook's **best model** (not a specific model name) was saved — the notebook auto-selects the lowest-RMSE model |
