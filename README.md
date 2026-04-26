# NY Grid Energy Demand Forecasting Dashboard

A data pipeline and forecasting model for NY-ISO electricity demand, with a Streamlit dashboard.  
Built as a portfolio project targeting quantitative analyst roles in energy trading.

---

## What this is

This project pulls real hourly electricity demand data from the EIA (Energy Information Administration),
stores it in a local SQLite database, trains a forecasting model, and displays everything in a live dashboard.

The NY-ISO region is the grid operator for New York state — the same market that FitzPatrick Nuclear Power Plant
dispatches into. Understanding demand patterns in this region is directly relevant to how Constellation's
commercial team in Baltimore thinks about hedging and pricing.

---

## Project structure

```
pjm_dashboard/
├── fetch_data.py     # Step 1: Pull data from EIA API → SQLite
├── model.py          # Step 2: Feature engineering + model training
├── dashboard.py      # Step 3: Streamlit visualization dashboard
├── energy_data.db    # Auto-created SQLite database
├── model.pkl         # Auto-saved trained model
├── scaler.pkl        # Auto-saved feature scaler
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install requests pandas numpy scikit-learn streamlit
```

### 2. Get a free EIA API key (30 seconds)

Go to: https://www.eia.gov/opendata/register.php  
Paste your key into `fetch_data.py` at the top: `API_KEY = "your_key_here"`

> **No key yet?** The pipeline runs with generated sample data by default — 
> you can still see the full dashboard and model working immediately.

### 3. Fetch data

```bash
python fetch_data.py
```

Pulls ~6 months of hourly demand data and saves to `energy_data.db`.

### 4. Train the model

```bash
python model.py
```

Builds features, compares Ridge Regression vs Gradient Boosting, saves best model.  
Prints accuracy metrics (MAE, RMSE, MAPE) to terminal.

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at http://localhost:8501

---

## What the model does

**Input features:**
- Hour of day (cyclically encoded as sin/cos)
- Day of week, weekend flag, business hours flag
- Month (cyclically encoded)
- Lag features: demand 1 hour ago, 24 hours ago, 168 hours ago (same time last week)
- Rolling statistics: 24-hour mean/std, 7-day mean

**Target:** Hourly electricity demand (MWh)

**Models compared:** Ridge Regression, Gradient Boosting Regressor  
**Evaluation:** 80/20 chronological train/test split, MAE / RMSE / MAPE

---

## Why this matters (energy context)

FitzPatrick Nuclear Power Plant operates as a baseload generator (~854 MW, nearly 24/7).
It sells electricity into the NY-ISO market at the real-time Locational Marginal Price (LMP).

When demand is high → LMP spikes → FitzPatrick earns more revenue.  
When demand is low → LMP drops → sometimes goes negative on windy spring nights.

Constellation's commercial/trading team in Baltimore hedges this exposure using 
day-ahead contracts. Accurately forecasting demand is core to that hedging strategy —
this is what the Risk Analytics and Quantitative Analyst teams work on daily.

---

## Next steps / extensions

- [ ] Add EIA API key and pull real data
- [ ] Pull actual LMP data from PJM Data Miner (requires free PJM account)
- [ ] Add weather data as a feature (temperature is the strongest demand driver)
- [ ] Build a simple Value-at-Risk (VaR) calculator based on price volatility
- [ ] Deploy to Streamlit Cloud for a live public URL

---

## Skills demonstrated

- REST API integration (EIA)
- SQLite database design and querying
- Pandas time-series manipulation
- Feature engineering (lag features, cyclical encoding, rolling statistics)
- Scikit-learn model training and evaluation
- Streamlit dashboard development
- Energy market domain knowledge

---

*Built by Dylan Pettinelli — portfolio project for quantitative analyst roles in energy trading*
"# ninemile-lmp-dashboard" 
