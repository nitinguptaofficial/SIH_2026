# AI Irrigation Optimizer

An ML-powered Flask web application that recommends irrigation amount and pump runtime from weather, soil, crop, and field conditions.

The project contains the trained scikit-learn pipeline, the source notebook, reference datasets, a server-rendered web interface, and a JSON prediction endpoint.

## Features

- Irrigation decision: `NO`, `YES / MODERATE`, or `YES / HIGH`
- Recommended water volume in litres
- Recommended pump runtime in minutes
- Model confidence estimate
- Automatic ET0 estimation from temperature and humidity
- HTML form interface and JSON API
- Input validation with readable error messages

## Repository Layout

```text
.
├── assets/
│   ├── data/                 Reference and training datasets
│   └── model/                Trained irrigation_model.joblib
├── docs/
│   ├── INSTRUCTIONS.md       Original project requirements
│   └── notebooks/            Source training notebook
├── src/
│   ├── app.py                Flask routes and request validation
│   ├── logic.py              Model loading and recommendation engine
│   ├── train_model.py        Optional model training script
│   ├── static/style.css      Application styles
│   └── templates/index.html  Web interface
├── render.yaml               Render deployment blueprint
├── requirements.txt          Python dependencies
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the development server

```bash
flask --app src.app run --debug
```

Open <http://127.0.0.1:5000/> in your browser.

The bundled model is already located at `assets/model/irrigation_model.joblib`. If you replace it, use the model produced by the source notebook and keep the feature schema unchanged.

## Deploy To Render

Render is the recommended deployment target for this Flask application.

### Option A: Blueprint deployment

1. Push the repository to GitHub.
2. Sign in at <https://render.com/> and select **New +** then **Blueprint**.
3. Connect the GitHub repository and select the branch to deploy.
4. Render detects `render.yaml` and uses:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --chdir src app:app`
   - Health check: `/`
5. Select **Apply**. When deployment finishes, open the generated `.onrender.com` URL.

### Option B: Manual web service

Create a Render **Web Service** from the GitHub repository and set:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn --chdir src app:app` |
| Health check path | `/` |

No environment variables are required for the current application.

## API Usage

The `POST /predict` endpoint returns JSON when the request includes `Accept: application/json`.

```bash
curl -X POST https://<your-render-service>.onrender.com/predict \
  -H "Accept: application/json" \
  -d "temperature_c=42&humidity_pct=30&wind_speed_kmph=4&rainfall_forecast_mm=0&soil_moisture_pct=20&field_capacity_pct=33&crop_type=Rice&soil_type=Loamy&growth_stage=Mid-season&field_area_m2=2000&flow_rate_lpm=200"
```

The response includes `irrigation_required`, `need_level`, `reason`, `raw_model_prediction_mm`, `recommended_water_liters`, `recommended_time_minutes`, and `confidence_pct`.

## Notebook And Model

The source notebook is stored at `docs/notebooks/AI_Irrigation_Optimizer_v4_larger_real_data.ipynb`.

The application does not retrain the model at startup. To retrain intentionally:

```bash
python src/train_model.py
```

This reads datasets from `assets/data/` and writes the trained artifact to `assets/model/irrigation_model.joblib`.

The model artifact must be compatible with the installed scikit-learn version. For production, pin the exact versions used to create the artifact if a model-loading compatibility warning appears.

## Demo Input

The notebook demo can be entered in the web form with these values:

| Input | Value |
|---|---:|
| Temperature | 42 °C |
| Relative humidity | 30% |
| Wind speed | 4 km/h |
| Rainfall forecast | 0 mm |
| Soil moisture | 20% |
| Field capacity | 33% |
| Crop | Rice |
| Soil type | Loamy |
| Growth stage | Mid-season |
| Field area | 2000 m² |
| Flow rate | 200 L/min |

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
