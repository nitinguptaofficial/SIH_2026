"""
app.py — Flask entry-point for AI Irrigation Optimizer
GET  /         → renders index.html with empty form
POST /predict  → reads form, calls recommend_irrigation(), shows result
               → returns JSON if Accept: application/json
"""

from flask import Flask, render_template, request, jsonify
import logic

app = Flask(__name__)

# Expose model availability to templates via context processor
@app.context_processor
def inject_model_status():
    return {
        "model_loaded": logic.MODEL_LOADED,
        "model_error":  logic.MODEL_ERROR,
    }


# ---------- helpers ----------------------------------------------------------

def _float(val, default=0.0):
    """Safe float conversion."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _validate(form):
    """Return (reading_dict, errors_list).  errors is empty on success."""
    errors = []

    def get_float(key, label, min_val=0.0, max_val=None):
        raw = form.get(key, "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            return None
        try:
            v = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if v < min_val:
            errors.append(f"{label} must be ≥ {min_val}.")
            return None
        if max_val is not None and v > max_val:
            errors.append(f"{label} must be ≤ {max_val}.")
            return None
        return v

    temperature_c        = get_float("temperature_c",        "Temperature (°C)",      -50, 60)
    humidity_pct         = get_float("humidity_pct",         "Humidity (%)",            0, 100)
    wind_speed_kmph      = get_float("wind_speed_kmph",      "Wind Speed (km/h)",       0)
    rainfall_forecast_mm = get_float("rainfall_forecast_mm", "Rainfall Forecast (mm)",  0)
    soil_moisture_pct    = get_float("soil_moisture_pct",    "Soil Moisture (%)",       0, 100)
    field_capacity_pct   = get_float("field_capacity_pct",   "Field Capacity (%)",      0, 100)
    field_area_m2        = get_float("field_area_m2",        "Field Area (m²)",         1)
    flow_rate_lpm        = get_float("flow_rate_lpm",        "Flow Rate (L/min)",       1)

    crop_type    = form.get("crop_type", "").strip()
    soil_type    = form.get("soil_type", "").strip()
    growth_stage = form.get("growth_stage", "").strip()

    if not crop_type:
        errors.append("Crop type is required.")
    if not soil_type:
        errors.append("Soil type is required.")
    if not growth_stage:
        errors.append("Growth stage is required.")

    if errors:
        return None, None, None, errors

    # ET0: auto-calculate from temperature + humidity (don't burden user)
    et0_mm = logic.estimate_et0(temperature_c, humidity_pct)

    reading = {
        "temperature_c":        temperature_c,
        "humidity_pct":         humidity_pct,
        "wind_speed_kmph":      wind_speed_kmph,
        "rainfall_forecast_mm": rainfall_forecast_mm,
        "soil_moisture_pct":    soil_moisture_pct,
        "field_capacity_pct":   field_capacity_pct,
        "et0_mm":               et0_mm,
        "crop_type":            crop_type,
        "soil_type":            soil_type,
        "growth_stage":         growth_stage,
    }
    return reading, float(field_area_m2), float(flow_rate_lpm), []


# ---------- routes -----------------------------------------------------------

@app.get("/")
def index():
    crops  = list(logic.KC_TABLE.keys())
    soils  = ["Sandy", "Loamy", "Clayey"]
    stages = ["Initial", "Development", "Mid-season", "Late-season"]
    return render_template("index.html", crops=crops, soils=soils, stages=stages)


@app.post("/predict")
def predict():
    crops  = list(logic.KC_TABLE.keys())
    soils  = ["Sandy", "Loamy", "Clayey"]
    stages = ["Initial", "Development", "Mid-season", "Late-season"]

    if not logic.MODEL_LOADED:
        error_msg = logic.MODEL_ERROR or "Model not loaded."
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": error_msg}), 503
        return render_template(
            "index.html",
            crops=crops, soils=soils, stages=stages,
            errors=[error_msg],
            form_data=request.form,
        )

    reading, field_area_m2, flow_rate_lpm, errors = _validate(request.form)

    if errors:
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"errors": errors}), 400
        return render_template(
            "index.html",
            crops=crops, soils=soils, stages=stages,
            errors=errors,
            form_data=request.form,
        )

    result = logic.recommend_irrigation(
        logic.pipeline, reading,
        field_area_m2=field_area_m2,
        flow_rate_lpm=flow_rate_lpm,
    )
    result["et0_mm_used"] = round(reading["et0_mm"], 3)

    if request.accept_mimetypes.best == "application/json":
        return jsonify(result)

    return render_template(
        "index.html",
        crops=crops, soils=soils, stages=stages,
        result=result,
        form_data=request.form,
    )


if __name__ == "__main__":
    app.run(debug=True)
