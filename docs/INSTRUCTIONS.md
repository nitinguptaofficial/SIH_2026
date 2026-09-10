# AI Irrigation Optimizer — Web App Build Instructions

## Context
You are given a Jupyter notebook (`docs/notebooks/AI_Irrigation_Optimizer_v4_larger_real_data.ipynb`) that trains a scikit-learn regression model to recommend irrigation amounts for a farm field. Your job is to turn this into a working, simple, locally-runnable **Flask web application** with a form-based frontend.

## What's in the notebook (source of truth)
- A trained `Pipeline` (preprocessor + regressor) saved via `joblib` to `artifacts/irrigation_model.joblib`
- Reference tables: `KC_TABLE`, `ROOT_ZONE_DEPTH_MM`, `CROP_SYNONYMS`, `SOIL_SYNONYMS`
- Helper functions: `estimate_et0()`, `normalize_crop()`, `normalize_soil()`, `get_confidence()`, `recommend_irrigation()`
- Feature schema: `feature_cols_num` and `feature_cols_cat` (defined in the notebook's feature engineering section)

## Task
Extract the model logic out of the notebook into a standalone Python module, then build a minimal Flask app around it.

### 1. Project structure to create
```
irrigation_app/
├── app.py
├── logic.py
├── assets/model/
│   └── irrigation_model.joblib      # copy from notebook's artifacts/ output
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```

### 2. `logic.py`
- Port over: `KC_TABLE`, `ROOT_ZONE_DEPTH_MM`, `CROP_SYNONYMS`, `SOIL_SYNONYMS`, `DEFAULT_KC`, `DEFAULT_ROOT_DEPTH`
- Port over: `normalize_crop()`, `normalize_soil()`, `estimate_et0()`, `get_confidence()`, `recommend_irrigation()`
- Load the trained pipeline once at import time: `pipeline = joblib.load("assets/model/irrigation_model.joblib")`
- Keep `feature_cols_num` / `feature_cols_cat` exactly as defined in the notebook — the model was trained on that exact column order/schema.

### 3. `app.py` (Flask)
- `GET /` → renders `index.html` with a form
- `POST /predict` → reads form fields, builds the `reading` dict expected by `recommend_irrigation()`, calls it, renders `index.html` again with the result displayed (or returns JSON if `Accept: application/json`)
- Form fields needed (match notebook feature schema): `temperature_c`, `humidity_pct`, `wind_speed_kmph`, `rainfall_forecast_mm`, `soil_moisture_pct`, `field_capacity_pct`, `et0_mm` (or auto-calculate via `estimate_et0()` if you'd rather not ask the user for it directly), `crop_type` (dropdown, from `KC_TABLE.keys()`), `soil_type` (dropdown: Sandy/Loamy/Clayey), `growth_stage` (dropdown: Initial/Development/Mid-season/Late-season)
- Also accept `field_area_m2` and `flow_rate_lpm` as optional inputs (default 1000 and 200) since `recommend_irrigation()` uses them to compute liters/minutes.
- Basic input validation: numeric fields must be non-negative; show a friendly error on bad input, don't crash.

### 4. `templates/index.html`
- Simple HTML form, one page, no JS framework needed (plain HTML + minimal vanilla JS is fine if you want live validation)
- Group inputs logically: Weather (temp/humidity/wind/rain), Soil (moisture/field capacity/soil type), Crop (crop type/growth stage), Field setup (area/flow rate)
- After submit, show the result clearly:
  - Irrigation Required: YES/NO (color-coded: green=NO/skip, yellow=MODERATE, red=HIGH)
  - Need level (LOW/MODERATE/HIGH)
  - Reason (plain text)
  - Recommended water (liters)
  - Recommended time (minutes)
  - Confidence (%)
- Keep styling simple and clean (`static/style.css`) — not required to be fancy, just readable and mobile-friendly.

### 5. `requirements.txt`
```
flask
scikit-learn
pandas
numpy
joblib
```
(Pin versions to match whatever scikit-learn version the notebook used, to avoid joblib load errors from version mismatches — check the notebook's pip environment if version info is available, otherwise use recent stable versions and note this as a known risk.)

### 6. `README.md`
Include: how to install (`pip install -r requirements.txt`), how to run (`flask --app src.app run` or `python src/app.py`), and a note that `assets/model/irrigation_model.joblib` must be copied in manually from the notebook's `artifacts/` folder output before running.

## Important constraints
- Do NOT retrain the model — just load the existing `.joblib` file.
- Do NOT change the feature schema/column order — it must exactly match what the notebook's `ColumnTransformer` was fit on, or predictions will silently break.
- Keep it simple: no database, no user auth, no React — single Flask app, server-rendered HTML, runs locally with `flask run`.
- If `assets/model/irrigation_model.joblib` is missing at startup, fail with a clear error message telling the user to export it from the notebook first, rather than crashing with a raw traceback.
- Test the app by running it and submitting the notebook's own demo values (Section 9 of the notebook) — confirm the output matches what the notebook prints.

## Deliverable
A working Flask app in the structure above, runnable with `pip install -r requirements.txt && flask --app src.app run`, that reproduces `recommend_irrigation()`'s output through a web form.

---

## MASTER PROMPT (paste this to the coding agent along with the notebook file)

> I have a trained ML pipeline for an irrigation recommendation system, saved in a Jupyter notebook (attached: `AI_Irrigation_Optimizer_v4_larger_real_data.ipynb`). Please read the notebook end to end — especially the feature engineering section (`feature_cols_num`, `feature_cols_cat`), the reference tables (`KC_TABLE`, `ROOT_ZONE_DEPTH_MM`, synonym dicts), and the `recommend_irrigation()` decision-engine function — and build a simple Flask web application around it, following the attached `INSTRUCTIONS.md` exactly.
>
> Build the full project structure described in `INSTRUCTIONS.md`: `app.py`, `logic.py`, `templates/index.html`, `static/style.css`, `requirements.txt`, `README.md`. Extract the model logic faithfully — don't retrain the model, don't change the feature schema, and preserve the exact same decision-tier logic (NO/LOW, YES/MODERATE, YES/HIGH) and output fields (`irrigation_required`, `need_level`, `reason`, `raw_model_prediction_mm`, `recommended_water_liters`, `recommended_time_minutes`, `confidence_pct`).
>
> The frontend should be a single clean HTML form covering all the inputs the model needs (weather, soil, crop, field setup), and after submission should display the recommendation clearly with color-coded urgency (green/yellow/red).
>
> Once built, run the app and verify it against the notebook's own Section 9 demo input to confirm the outputs match. Then give me clear run instructions.
