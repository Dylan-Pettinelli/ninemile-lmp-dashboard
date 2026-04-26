"""
fetch_data.py
-------------
Pulls real LMP (Locational Marginal Price) data from PJM's Data Miner API
for the NINEMILE node in Oswego, NY — the grid node adjacent to FitzPatrick.

This is real, live data. PJM updates it hourly.

SETUP (do this once before running):
  1. Create a file called .env in this folder
  2. Add these two lines to it:
        PJM_USERNAME=your_pjm_username
        PJM_PASSWORD=your_pjm_password
  3. Make sure .env is in your .gitignore so it never goes to GitHub

Run with:  python fetch_data.py
"""

import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta

# ── Credential management ─────────────────────────────────────────────────────
def load_env():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            "\n[!] No .env file found. Create one in this folder with:\n"
            "    PJM_USERNAME=your_username\n"
            "    PJM_PASSWORD=your_password\n"
        )
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

load_env()

PJM_USERNAME = os.environ.get("PJM_USERNAME")
PJM_PASSWORD = os.environ.get("PJM_PASSWORD")
PJM_API_KEY = os.environ.get("PJM_API_KEY")


if not PJM_USERNAME or not PJM_PASSWORD:
    raise ValueError("[!] PJM_USERNAME or PJM_PASSWORD missing from .env file.")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = "energy_data.db"
PNODE_ID   = 1067164095
PNODE_NAME = "NINEMILE"
API_BASE   = "https://api.pjm.com/api/v1/rt_hrl_lmps"
WEATHER_BASE = "https://archive-api.open-meteo.com/v1/archive"

FIELDS = (
    "datetime_beginning_utc,"
    "datetime_beginning_ept,"
    "pnode_id,"
    "pnode_name,"
    "total_lmp_rt,"
    "system_energy_price_rt,"
    "congestion_price_rt,"
    "marginal_loss_price_rt"
)


# ── Database setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lmp_data (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime_utc            TEXT UNIQUE,
            datetime_ept            TEXT,
            pnode_id                INTEGER,
            pnode_name              TEXT,
            total_lmp_rt            REAL,
            system_energy_price_rt  REAL,
            congestion_price_rt     REAL,
            marginal_loss_price_rt  REAL,
            fetched_at              TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at      TEXT,
            records_added   INTEGER,
            start_date      TEXT,
            end_date        TEXT,
            status          TEXT,
            message         TEXT
        )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weather_data (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        datetime_ept TEXT UNIQUE,
        temp_f       REAL,
        fetched_at   TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database ready: {os.path.abspath(DB_PATH)}")


# ── PJM API ───────────────────────────────────────────────────────────────────
def fetch_lmp(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly real-time LMP data from PJM Data Miner for NINEMILE node.
    """
    params = {
        "sort":             "datetime_beginning_ept",
        "order":            "Asc",
        "startRow":         1,
        "isActiveMetadata": "true",
        "fields":           "congestion_price_rt,datetime_beginning_ept,datetime_beginning_utc,marginal_loss_price_rt,pnode_id,pnode_name,system_energy_price_rt,total_lmp_rt",
        "datetime_beginning_ept": f"{start_date} 00:00to{end_date}",
        "pnode_id":         PNODE_ID,
        "format":           "json",
        "download":         "true",
    }

    print(f"[API] Fetching LMP: {start_date} → {end_date} ...")

    response = requests.get(
        API_BASE,
        params=params,
        headers={"Ocp-Apim-Subscription-Key": PJM_API_KEY},
        timeout=30
    )

    if response.status_code == 401:
        raise PermissionError("[!] Auth failed. Check username/password in .env")
    if response.status_code == 403:
        raise PermissionError("[!] Access denied. Confirm PJM Public access on your account.")

    response.raise_for_status()

    data = response.json()
    records = data if isinstance(data, list) else data.get("items", [])

    if not records:
        print("[!] No records returned.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"[API] Received {len(df)} records.")

    df["datetime_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df["datetime_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    for col in ["total_lmp_rt", "system_energy_price_rt", "congestion_price_rt", "marginal_loss_price_rt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ── WEATHER ───────────────────────────────────────────────────────────────────

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":        43.45,
        "longitude":       -76.51,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          "temperature_2m",
        "timezone":        "America/New_York",
        "temperature_unit": "fahrenheit",
    }

    print(f"[WEATHER] Fetching temperature: {start_date} → {end_date} ...")

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    times  = data["hourly"]["time"]
    temps  = data["hourly"]["temperature_2m"]

    df = pd.DataFrame({"datetime_ept": times, "temp_f": temps})
    df["datetime_ept"] = pd.to_datetime(df["datetime_ept"])

    print(f"[WEATHER] Got {len(df)} hourly records.")
    return df

def save_weather_to_db(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    inserted = 0
    now = datetime.now().isoformat()

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO weather_data (datetime_ept, temp_f, fetched_at)
                VALUES (?, ?, ?)
            """, (
                str(row["datetime_ept"]),
                row["temp_f"],
                now
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return inserted

# ── Database operations ───────────────────────────────────────────────────────
def save_to_db(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    inserted = 0
    now = datetime.now().isoformat()

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO lmp_data (
                    datetime_utc, datetime_ept, pnode_id, pnode_name,
                    total_lmp_rt, system_energy_price_rt,
                    congestion_price_rt, marginal_loss_price_rt, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get("datetime_utc")),
                str(row.get("datetime_ept")),
                PNODE_ID,
                PNODE_NAME,
                row.get("total_lmp_rt"),
                row.get("system_energy_price_rt"),
                row.get("congestion_price_rt"),
                row.get("marginal_loss_price_rt"),
                now
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return inserted


def log_fetch(start: str, end: str, added: int, status: str, message: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO fetch_log (fetched_at, records_added, start_date, end_date, status, message)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), added, start, end, status, message))
    conn.commit()
    conn.close()


# ── Query helpers ─────────────────────────────────────────────────────────────
def load_from_db(days_back: int = 90) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

    df = pd.read_sql_query("""
        SELECT
            datetime_utc,
            datetime_ept,
            total_lmp_rt,
            system_energy_price_rt,
            congestion_price_rt,
            marginal_loss_price_rt
        FROM lmp_data
        WHERE datetime_utc >= ?
        ORDER BY datetime_utc ASC
    """, conn, params=(cutoff,))

    conn.close()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["datetime_ept"] = pd.to_datetime(df["datetime_ept"])
    return df


def get_db_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MIN(datetime_utc), MAX(datetime_utc) FROM lmp_data")
    count, earliest, latest = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM fetch_log WHERE status = 'success'")
    fetches = cursor.fetchone()[0]
    conn.close()
    return {"total_hours": count, "earliest": earliest, "latest": latest, "fetches_run": fetches}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Nine Mile LMP Fetcher | Node: {PNODE_NAME} ({PNODE_ID})")
    print("=" * 60)

    init_db()

    end   = datetime.now()
    start = end - timedelta(days=90)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    try:
        df = fetch_lmp(start_str, end_str)
        if not df.empty:
            added = save_to_db(df)
            log_fetch(start_str, end_str, added, "success")
            print(f"[DB] Saved {added} new records.")
        else:
            log_fetch(start_str, end_str, 0, "empty", "No data returned")
    except Exception as e:
        log_fetch(start_str, end_str, 0, "error", str(e))
        raise

    stats = get_db_stats()
    print("\n[STATS]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    weather_df = fetch_weather(start_str, end_str)
    weather_added = save_weather_to_db(weather_df)
    print(f"[DB] Saved {weather_added} weather records.")

    print("\nDone. Next: python model.py")


if __name__ == "__main__":
    main()