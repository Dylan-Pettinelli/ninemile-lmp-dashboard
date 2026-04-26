# Nine Mile Point — LMP Price Forecasting Dashboard

A real-time data pipeline and forecasting model for electricity prices at the **NINEMILE pricing node** in Oswego, NY — the grid interconnection point for Nine Mile Point Nuclear Power Plant, operated by Constellation Energy.

Built as a portfolio project targeting quantitative analyst and risk analytics roles in energy trading.

---

## What this is

This project pulls real hourly **Locational Marginal Price (LMP)** data from PJM's Data Miner API for the Nine Mile node, stores it in a local SQLite database, trains a price forecasting model, and displays everything in a live Streamlit dashboard.

LMP is the actual dollar price Constellation receives for each megawatt-hour generated at Nine Mile — the same number Constellation's commercial team in Baltimore watches and hedges against daily. This project models that price, analyzes congestion patterns, and forecasts the next 24 hours.

---

## Project structure

```
ninemile-lmp-dashboard/
├── fetch_data.py     # Pulls real LMP data from PJM Data Miner API → SQLite
├── model.py          # Feature engineering + forecasting model training
├── dashboard.py      # Streamlit dashboard (price history, congestion, forecast)
├── .gitignore        # Excludes .env, database, and model files
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install requests pandas numpy scikit-learn streamlit
```

### 2. Set up credentials

Create a `.env` file in the project folder:

```
PJM_USERNAME=your_pjm_username
PJM_PASSWORD=your_pjm_password
PJM_API_KEY=your_pjm_api_key
```

Getting access:
- Register at [accountmanager.pjm.com](https://accountmanager.pjm.com) — free, non-member account works
- Request **PJM Public** access during registration
- Find your API key in the Network tab of browser DevTools when using Data Miner

### 3. Fetch data

```bash
python fetch_data.py
```

Pulls 90 days of real hourly LMP data for the NINEMILE node (pnode\_id: 1067164095) and saves to a local SQLite database.

### 4. Train the model

```bash
python model.py
```

Engineers features, compares Ridge Regression vs Gradient Boosting, saves the best model. Prints MAE, RMSE to terminal.

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

## What the model does

Forecasts hourly LMP ($/MWh) for the next 24 hours using time-series features:

**Time features**
- Hour of day and month, encoded cyclically as sin/cos so hour 23 and hour 0 are treated as close together
- Day of week, weekend flag, business hours flag

**Lag features**
- LMP 1 hour ago, 24 hours ago, 168 hours ago (same hour last week)
- These are typically the strongest predictors in energy price time-series

**Rolling statistics**
- 24-hour rolling mean and standard deviation
- 7-day rolling mean

**Models compared:** Ridge Regression, Gradient Boosting Regressor  
**Train/test split:** 80/20 chronological — older data trains, most recent 20% tests  
**Evaluation metrics:** MAE and RMSE (in $/MWh)

Note: LMP forecasting is genuinely difficult. Prices spike unpredictably due to transmission events, weather, and generation outages. The model captures daily and weekly patterns well but will underestimate spikes — a known limitation of regression approaches on price data.

---

## Dashboard features

- **LMP price history** — Total LMP vs System Energy Price, showing the spread caused by congestion
- **Congestion analysis** — Hours of positive vs negative congestion, worst congestion events
- **24-hour forecast** — Model prediction overlaid on recent actuals
- **Price shape** — Average LMP by hour of day and day of week
- **Model performance** — Actual vs predicted chart with MAE and RMSE metrics

---

## Why this matters — energy market context

**Nine Mile Point** (Units 1 & 2) and **FitzPatrick Nuclear Power Plant** are adjacent baseload generators in Oswego, NY, both operated by Constellation Energy. They run at full output around the clock and sell every megawatt-hour into the PJM real-time market at the LMP for their grid node.

**LMP has three components:**
- **System energy price** — the grid-wide clearing price, set by the marginal generator (usually a gas plant)
- **Congestion price** — positive when transmission lines have spare capacity, negative when they're constrained. Sustained negative congestion means Nine Mile earns less than the grid-wide price because the lines out of Oswego can't carry all the power being generated
- **Marginal loss price** — small adjustment for electrical losses over long distances

**Why Constellation's Baltimore team cares:**  
The commercial/trading team hedges Nine Mile's exposure to LMP volatility using forward contracts, options, and Financial Transmission Rights (FTRs). Accurately forecasting LMP — and specifically congestion patterns at the Oswego node — informs decisions about how much output to hedge, at what price, and how far forward. This is the core function of the Risk Analytics and Quantitative Analyst teams.

---

## Data source

**PJM Data Miner API** — `api.pjm.com/api/v1/rt_hrl_lmps`  
Node: NINEMILE · pnode\_id: 1067164095 · Zone: ATSI · Updated hourly by PJM

---

## Planned improvements

- [ ] Add weather data (temperature) as a model feature — the single strongest driver of electricity demand and therefore LMP
- [ ] Pull day-ahead LMP alongside real-time to model the DA/RT spread
- [ ] Build a simple VaR (Value at Risk) calculator based on historical LMP volatility
- [ ] Deploy to Streamlit Cloud for a live public URL

---

*Built by Dylan Pettinelli — portfolio project for quantitative analyst roles in energy trading*