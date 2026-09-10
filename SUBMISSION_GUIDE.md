# Submission Guide

## Project

AI Irrigation Optimizer is a Flask web application that combines a trained scikit-learn pipeline with crop, soil, weather, and field information to recommend irrigation amount and runtime.

## Entry points

- Application: `src/app.py`
- Decision logic: `src/logic.py`
- Training notebook: `docs/notebooks/AI_Irrigation_Optimizer_v4_larger_real_data.ipynb`
- Model artifact: `assets/model/irrigation_model.joblib`

## Demo

Install dependencies from the repository root, then run `flask --app src.app run`. Open `http://127.0.0.1:5000/` and submit the sample values in the root README.

The `/predict` endpoint also supports JSON responses with `Accept: application/json`.
