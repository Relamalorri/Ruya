# Ruya — Coffee Sales Dashboard & ML Forecasting

An interactive sales analytics dashboard for coffee shop data, built with multiple machine learning forecasting models. Ruya allows users to explore historical sales trends and predict future demand using ARIMA, LSTM, and Prophet — side by side.

> Samsung Innovation Campus — Capstone Group Project
> Forked from AbAlowaid/Ruya

---

## Live Demo

[View Live Dashboard](https://naseralkuhili-ruya-app-ui-branding-overhaul-ibzxbl.streamlit.app/)

> The app may take a few seconds to wake up on first load (Streamlit free tier).

---

## The Problem

Coffee shop owners and analysts struggle to make inventory and staffing decisions without reliable sales forecasts. Existing tools are either too generic or require advanced technical knowledge. Ruya makes forecasting accessible through a clean, interactive dashboard.

---

## What Ruya Does

- Visualizes historical coffee sales trends interactively
- Forecasts future sales using 3 different ML models
- Allows side-by-side comparison of forecasting approaches
- Presents results in a clean, non-technical dashboard interface

---

## Pipeline

1. **Raw Sales Data** — Daily coffee sales records collected and loaded
2. **Data Preprocessing** — Handle missing values, parse dates, normalize features for LSTM, extract temporal components (day, week, month, seasonality)
3. **Exploratory Data Analysis** — Visualize daily, weekly, and monthly trends, identify seasonal patterns, detect outliers
4. **Model Training** — Three models trained independently: ARIMA on the time series, LSTM as a sequential neural network, Prophet with seasonality decomposition
5. **Model Evaluation** — Forecast accuracy compared across all 3 models, predicted vs actual sales visualized
6. **Dashboard Deployment** — Streamlit app serving all 3 models interactively, users select model and forecast horizon, results displayed with charts and metrics

---

## Forecasting Models

| Model | Type | Strength |
|---|---|---|
| ARIMA | Statistical time-series | Fast, interpretable, great for stable trends |
| LSTM | Deep learning (neural network) | Captures complex non-linear patterns |
| Prophet | ML (Meta/Facebook) | Handles seasonality and holidays well |

---

## Repository Structure

| Folder / File | Contents |
|---|---|
| `notebooks/Ruya.ipynb` | Full analysis, data exploration, and model training |
| `data/Coffe_sales.csv` | Raw coffee sales dataset |
| `data/coffeedata_prepped.csv` | Preprocessed and cleaned data |
| `models/arima_model.pkl` | Saved ARIMA model |
| `models/lstm_model.keras` | Saved LSTM model |
| `models/prophet_model.pkl` | Saved Prophet model |
| `models/lstm_scaler.pkl` | Scaler used for LSTM normalization |
| `app.py` | Streamlit dashboard application |
| `requirements.txt` | Python dependencies |

---

## How to Run Locally

1. Clone the repo: `git clone https://github.com/Relamalorri/Ruya.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`

---

## Tech Stack

Python · Pandas · Streamlit · ARIMA (statsmodels) · LSTM (Keras/TensorFlow) · Prophet (Meta) · Scikit-learn · Jupyter Notebook

---

## Dataset

Coffee shop daily sales data used for time-series analysis and forecasting. Preprocessed to handle missing values, normalize features, and extract temporal components including day, week, month, and seasonality patterns.
