"""
dashboard.py
------------
Streamlit dashboard for Nine Mile LMP Price Forecasting.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nine Mile LMP Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH    = "energy_data.db"
MODEL_PATH = "model.pkl"


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(days_back: int = 90) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
    df = pd.read_sql_query("""
        SELECT
            datetime_utc        AS timestamp,
            total_lmp_rt        AS lmp,
            system_energy_price_rt AS system_price,
            congestion_price_rt AS congestion,
            marginal_loss_price_rt AS marginal_loss
        FROM lmp_data
        WHERE datetime_utc >= ?
        ORDER BY datetime_utc ASC
    """, conn, params=(cutoff,))
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=300)
def get_forecast() -> pd.DataFrame:
    try:
        from model import forecast_next_24h
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT datetime_utc AS timestamp, total_lmp_rt AS demand_mwh FROM lmp_data ORDER BY datetime_utc ASC",
            conn
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return forecast_next_24h(df)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


@st.cache_data(ttl=300)
def get_model_performance() -> dict:
    try:
        from model import build_features, FEATURE_COLS, TARGET_COL
        import pickle

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT datetime_utc AS timestamp, total_lmp_rt AS demand_mwh FROM lmp_data ORDER BY datetime_utc ASC",
            conn
        )
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        featured = build_features(df)

        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        model      = model_data["model"]
        model_name = model_data["name"]

        split_idx = max(0, len(featured) - 720)
        test_df   = featured.iloc[split_idx:].copy()
        X_test    = test_df[FEATURE_COLS].values

        if model_name == "Ridge Regression":
            X_test = scaler.transform(X_test)

        preds   = model.predict(X_test)
        actuals = test_df[TARGET_COL].values

        mae  = np.mean(np.abs(actuals - preds))
        mape = np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))

        return {
            "mae": mae, "mape": mape, "rmse": rmse,
            "model_name": model_name,
            "timestamps": test_df["timestamp"].values,
            "actuals":    actuals,
            "preds":      preds,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Nine Mile LMP")
    st.markdown("---")
    st.markdown("**Data source:** PJM Data Miner API")
    st.markdown("**Node:** NINEMILE (pnode 1067164095)")
    st.markdown("**Location:** Oswego, NY")
    st.markdown("**Zone:** ATSI")
    st.markdown("---")

    days_back = st.slider("Days of history", min_value=7, max_value=90, value=30)

    st.markdown("---")
    st.markdown("### About this project")
    st.markdown("""
    Real-time LMP price data for the Nine Mile Point nuclear node
    in Oswego, NY — adjacent to FitzPatrick Nuclear Power Plant,
    both operated by Constellation Energy.

    The model forecasts hourly LMP using time, lag, and rolling
    features — the same core approach used by commercial risk
    analytics teams.

    **Stack:** Python · PJM API · SQLite · scikit-learn · Streamlit

    Built by **Dylan Pettinelli**
    """)

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("⚡ Nine Mile Point — Real-Time LMP Dashboard")
st.caption(f"NINEMILE node · PJM Data Miner · Updated {datetime.now().strftime('%b %d, %Y %I:%M %p')}")

df = load_data(days_back)

if df.empty:
    st.error("No data found. Run `python fetch_data.py` first.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

latest_lmp  = df["lmp"].iloc[-1]
avg_lmp     = df["lmp"].mean()
peak_lmp    = df["lmp"].max()
peak_time   = df.loc[df["lmp"].idxmax(), "timestamp"]
avg_cong    = df["congestion"].mean()

mid         = len(df) // 2
prior_avg   = df["lmp"].iloc[:mid].mean()
recent_avg  = df["lmp"].iloc[mid:].mean()
delta_pct   = ((recent_avg - prior_avg) / prior_avg) * 100

col1.metric(
    "Current LMP",
    f"${latest_lmp:,.2f}/MWh",
    delta=f"${latest_lmp - avg_lmp:+,.1f} vs avg"
)
col2.metric(
    "Avg LMP",
    f"${avg_lmp:,.2f}/MWh",
    delta=f"{delta_pct:+.1f}% vs prior period"
)
col3.metric(
    "Period Peak",
    f"${peak_lmp:,.2f}/MWh",
    delta=peak_time.strftime("%b %d, %I%p")
)
col4.metric(
    "Avg Congestion",
    f"${avg_cong:+,.2f}/MWh",
    delta="negative = lines congested",
    delta_color="off"
)

st.markdown("---")

# ── LMP chart ─────────────────────────────────────────────────────────────────
st.subheader("📈 Hourly LMP — NINEMILE Node ($/MWh)")

if days_back > 30:
    display_df = df.set_index("timestamp")[["lmp", "system_price", "congestion"]].resample("6h").mean().reset_index()
    grain = "6-hour avg"
else:
    display_df = df.copy()
    grain = "hourly"

st.caption(f"{grain} resolution · Total LMP = System Energy Price + Congestion + Marginal Loss")
st.line_chart(
    display_df.set_index("timestamp")[["lmp", "system_price"]].rename(
        columns={"lmp": "Total LMP ($/MWh)", "system_price": "System Energy Price ($/MWh)"}
    ),
    use_container_width=True
)

# ── Two columns ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🔮 24-Hour Ahead Forecast")
    model_exists = os.path.exists(MODEL_PATH)

    if not model_exists:
        st.warning("No model found. Run `python model.py` first.")
    else:
        with st.spinner("Generating forecast..."):
            forecast_df = get_forecast()

        if "error" in forecast_df.columns:
            st.error(f"Forecast error: {forecast_df['error'].iloc[0]}")
        else:
            recent_actuals  = df.tail(48)[["timestamp", "lmp"]].rename(columns={"lmp": "Actual LMP"})
            forecast_display = forecast_df.rename(columns={"forecast_mwh": "Forecast LMP"})

            combined = pd.merge(
                recent_actuals.set_index("timestamp"),
                forecast_display.set_index("timestamp"),
                left_index=True, right_index=True, how="outer"
            )
            st.line_chart(combined, use_container_width=True)
            st.caption("Actual LMP (blue) vs 24-hour model forecast (orange) · $/MWh")

            with st.expander("View forecast table"):
                forecast_df["hour"] = pd.to_datetime(forecast_df["timestamp"]).dt.strftime("%a %b %d %I%p")
                st.dataframe(
                    forecast_df[["hour", "forecast_mwh"]].rename(
                        columns={"hour": "Hour", "forecast_mwh": "Forecast LMP ($/MWh)"}
                    ),
                    hide_index=True, use_container_width=True
                )

with col_right:
    st.subheader("🕐 Avg LMP by Hour of Day")
    st.caption("Daily price shape — when LMP is highest and lowest")

    hourly = (
        df.assign(hour=df["timestamp"].dt.hour)
        .groupby("hour")["lmp"]
        .mean()
        .reset_index()
        .rename(columns={"hour": "Hour", "lmp": "Avg LMP ($/MWh)"})
    )
    st.bar_chart(hourly.set_index("Hour"), use_container_width=True)

    st.subheader("📅 Avg LMP by Day of Week")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = (
        df.assign(dow=df["timestamp"].dt.dayofweek)
        .groupby("dow")["lmp"]
        .mean()
        .reset_index()
    )
    dow["day"] = dow["dow"].map(dict(enumerate(day_names)))
    st.bar_chart(dow.set_index("day")["lmp"], use_container_width=True)


# ── Congestion analysis ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔌 Congestion Analysis")
st.caption("Congestion price = difference between what the grid clears at vs what Nine Mile actually receives. Negative = transmission lines out of Oswego are constrained.")

cong_df = df[["timestamp", "congestion"]].copy()
if days_back > 30:
    cong_display = cong_df.set_index("timestamp").resample("6h").mean().reset_index()
else:
    cong_display = cong_df.copy()

st.line_chart(cong_display.set_index("timestamp")["congestion"].rename("Congestion Price ($/MWh)"), use_container_width=True)

c1, c2, c3 = st.columns(3)
neg_hours = (df["congestion"] < 0).sum()
pos_hours = (df["congestion"] > 0).sum()
worst_cong = df["congestion"].min()
worst_time = df.loc[df["congestion"].idxmin(), "timestamp"]

c1.metric("Hours w/ Negative Congestion", f"{neg_hours}", delta=f"{neg_hours/len(df)*100:.0f}% of period")
c2.metric("Hours w/ Positive Congestion", f"{pos_hours}", delta=f"{pos_hours/len(df)*100:.0f}% of period")
c3.metric("Worst Congestion", f"${worst_cong:,.2f}/MWh", delta=worst_time.strftime("%b %d, %I%p"))


# ── Model performance ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Performance (last 30 days)")
st.caption("How well the forecasting model predicted actual LMP. MAE = average dollar miss per hour.")

model_exists = os.path.exists(MODEL_PATH)
if model_exists:
    with st.spinner("Evaluating..."):
        perf = get_model_performance()

    if "error" in perf:
        st.error(f"Error: {perf['error']}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"${perf['mae']:,.2f}/MWh",
                  help="Average absolute forecast error in $/MWh")
        m2.metric("RMSE", f"${perf['rmse']:,.2f}/MWh",
                  help="Root Mean Squared Error — penalizes large misses more heavily")
        m3.metric("Model", perf["model_name"])

        perf_df = pd.DataFrame({
            "timestamp": perf["timestamps"],
            "Actual LMP":    perf["actuals"],
            "Predicted LMP": perf["preds"],
        }).set_index("timestamp")
        perf_display = perf_df.resample("6h").mean()
        st.line_chart(perf_display, use_container_width=True)
        st.caption("Actual vs predicted LMP (6-hour averages) · $/MWh · last 30 days")
else:
    st.info("Run `python model.py` to see model performance here.")


# ── Context panel ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚛️ Why This Data Matters — Nine Mile, FitzPatrick & Constellation's Commercial Team"):
    st.markdown("""
    ### The commercial picture

    **Nine Mile Point** (Units 1 & 2) and **FitzPatrick** sit within a mile of each other
    in Oswego, NY — both operated by Constellation Energy. They are baseload nuclear generators,
    meaning they run at full output ~24/7 regardless of market conditions.

    Every megawatt-hour they produce gets sold into the PJM real-time market at the **LMP for
    their grid node** — the price you're looking at in this dashboard.

    ### What drives the spikes you see

    - **Cold snaps / heat waves:** Demand surges, gas peakers set a high marginal price, LMP spikes
    - **Negative congestion:** Too much power trying to flow out of Oswego on constrained transmission
      lines — Nine Mile gets paid *less* than the grid-wide clearing price
    - **Overnight lows:** Baseload nuclear still runs, but demand is low and LMP drops

    ### What Constellation's Baltimore team does with this

    The commercial/trading team hedges Constellation's exposure to LMP volatility using:
    - **Forward contracts:** Lock in a fixed $/MWh price for future output
    - **FTRs (Financial Transmission Rights):** Hedge specifically against congestion losses
    - **Options:** Buy the right to sell at a floor price without capping upside

    A **quant analyst** in Baltimore builds models like this one to forecast LMP,
    quantify risk exposure, and inform hedging decisions — determining how much output
    to hedge, at what price, and how far forward.

    ### The congestion story

    Watch the congestion chart above. Sustained negative congestion (lines out of Oswego
    constrained) directly reduces Nine Mile's revenue relative to the system price.
    Constellation's traders buy FTRs to offset this — and the decision of how many to buy
    depends on forecasting how often and how severely congestion will occur.
    """)

st.markdown("---")
st.caption("Built with Python · PJM Data Miner API · SQLite · scikit-learn · Streamlit | Dylan Pettinelli | Portfolio Project")