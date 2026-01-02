import os
import sys
import numpy as np
import pandas as pd
import json
import django
from datetime import datetime
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

from django.conf import settings
from API.models import Station, PollutantData, MeteorologicalData
from API.models import MapView

BASE_DIR = settings.BASE_DIR

DATA_DIR_ENV = os.getenv("Data_Dir")

if DATA_DIR_ENV:
    INPUT_DIR = DATA_DIR_ENV
else:
    INPUT_DIR = os.path.join(BASE_DIR, "Dataset", "example")
    
OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset", "Preprocess_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_float(val):
    try:
        val = float(val)
        if np.isnan(val):
            return None
        return val
    except:
        return None


def extract_short_code(col_name: str):
    known_codes = ["tn", "tx", "tavg", "rhavg", "rr", "ffx", "ffavg"]

    raw = col_name.strip().lower()

    if "(" in raw and ")" in raw:
        inside = raw.split("(", 1)[1].split(")")[0].strip()
        if inside in known_codes:
            return inside

    base = raw.split("(", 1)[0].strip()
    cleaned = "".join(ch for ch in base if ch.isalpha())

    for code in known_codes:
        if cleaned == code:
            return code

    for code in known_codes:
        if code in raw:
            return code

    mapping = {
        "temperaturminimum": "tn",
        "temperaturmaksimum": "tx",
        "temperaturratarata": "tavg",
        "kelembapanratarata": "rhavg",
        "kelembapanrata": "rhavg",
        "curahhujan": "rr",
        "kecepatananginmaksimum": "ffx",
        "kecepatananginrata": "ffavg",
    }

    for long_word, short in mapping.items():
        if long_word in raw:
            return short

    return cleaned[:5] or raw


METEO_FIELD_MAP = {
    "tn": "temperatur_minimum",
    "tx": "temperatur_maksimum",
    "tavg": "temperatur_rata",
    "rhavg": "kelembapan_rata",
    "rr": "curah_hujan",
    "ffx": "kecepatan_angin_maksimum",
    "ffavg": "kecepatan_angin_rata",
}

def insert_meteorologi_row(row, station):
    data = {}

    for col in row.index:
        short = extract_short_code(col)

        if short in METEO_FIELD_MAP:
            mapped_field = METEO_FIELD_MAP[short]
            data[mapped_field] = safe_float(row.get(col))

    tanggal = pd.to_datetime(row.get("Tanggal"), errors="coerce")
    if pd.isna(tanggal):
        return False

    MeteorologicalData.objects.update_or_create(
        station=station,
        tanggal=tanggal.date(),
        defaults=data
    )
    return True

def insert_polutan_row(row, station):
    tanggal = pd.to_datetime(row.get("Tanggal"), errors="coerce")
    if pd.isna(tanggal):
        return False

    PollutantData.objects.update_or_create(
        station=station,
        tanggal=tanggal.date(),
        defaults={
            "pm10": safe_float(row.get("pm10")),
            "pm25": safe_float(row.get("pm25")),
            "so2":  safe_float(row.get("so2")),
            "co":   safe_float(row.get("co")),
            "o3":   safe_float(row.get("o3")),
            "no2":  safe_float(row.get("no2")),
        }
    )
    return True

def clean_data(df):
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df.replace(["-", "–", "N/A", "na", "NaN", 8888, 9999, ""], np.nan, inplace=True)
    df.columns = df.columns.str.strip()
    return df

def fill_missing_window(df, window=15):
    df = df.copy()
    for col in df.columns:
        if col.lower() in ["tanggal", "stasiun", "station"]:
            continue

        series = df[col].astype(float)

        for i in range(len(series)):
            if pd.isna(series.iloc[i]):
                start = max(0, i - window)
                end = min(len(series), i + window)
                median_val = np.nanmedian(series.iloc[start:end])
                if not np.isnan(median_val):
                    series.iloc[i] = median_val

        df[col] = series
    return df

def remove_outliers(df):
    df = df.copy()

    for col in df.columns:
        if col.lower() in ["tanggal", "stasiun", "station"]:
            continue

        x = df[col].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        x[(x < low) | (x > up)] = np.nan
        df[col] = x

    return fill_missing_window(df)

def detect_file_type(df):
    cols = [c.lower() for c in df.columns]

    if "pm10" in cols or "pm25" in cols:
        return "polutan"

    for code in METEO_FIELD_MAP.keys():
        if code in cols:
            return "meteo"

    for key in ["temperatur", "kelembapan", "curah", "angin"]:
        if any(key in c for c in cols):
            return "meteo"

    return None

def calculate_ispu_dominant(row):
    polutans = ["pm10", "pm25", "so2", "co", "o3", "no2"]
    values = {p: row.get(p) for p in polutans if row.get(p) is not None}

    if not values:
        return "Tidak Ada Data"

    dominant = max(values, key=values.get)
    return dominant.upper()

def update_mapview_last_row(df, station):
    if df.empty or "Tanggal" not in df.columns:
        return

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df = df[df["Tanggal"].notna()]

    if df.empty:
        return

    last = df.sort_values("Tanggal").iloc[-1]

    pm10 = safe_float(last.get("pm10"))
    pm25 = safe_float(last.get("pm25"))
    so2 = safe_float(last.get("so2"))
    co = safe_float(last.get("co"))
    o3 = safe_float(last.get("o3"))
    no2 = safe_float(last.get("no2"))

    ispu = calculate_ispu_dominant({
        "pm10": pm10,
        "pm25": pm25,
        "so2": so2,
        "co": co,
        "o3": o3,
        "no2": no2,
    })

    MapView.objects.update_or_create(
        station=station,
        tanggal=last["Tanggal"].date(),
        defaults={
            "pm10": pm10,
            "pm25": pm25,
            "so2": so2,
            "co": co,
            "o3": o3,
            "no2": no2,
            "indeks_kualitas_udara": ispu,
        }
    )

    print(f"[MAPVIEW UPDATED] {station.nama_stasiun} — "
          f"{last['Tanggal'].date()} — ISPU={ispu}")
    
def get_last_polutan_row(df, station_name=None):
    if df.empty:
        return None

    # pastikan kolom Tanggal benar
    if "tanggal" in df.columns and "Tanggal" not in df.columns:
        df = df.rename(columns={"tanggal": "Tanggal"})

    if "Tanggal" not in df.columns:
        return None

    df = df.copy()
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df = df[df["Tanggal"].notna()]
    if df.empty:
        return None

    # ambil baris terbaru
    last = df.sort_values("Tanggal").iloc[-1]

    return {
        "Station": station_name,  # kalau ga mau, boleh hapus
        "Tanggal": last["Tanggal"].date().isoformat(),
        "pm10": safe_float(last.get("pm10")),
        "pm25": safe_float(last.get("pm25")),
        "so2": safe_float(last.get("so2")),
        "co": safe_float(last.get("co")),
        "o3": safe_float(last.get("o3")),
        "no2": safe_float(last.get("no2")),
    }


def preprocess_uploaded_df(df, station=None, file_type=None):
    df = clean_data(df)
    df = fill_missing_window(df)
    df = remove_outliers(df)

    if station is not None and file_type == "polutan":
        update_mapview_last_row(df, station)

    return df

if __name__ == "__main__":
    print("==== PREPROCESS_1 DIMULAI ====")
    print("Input :", INPUT_DIR)
    print("Output:", OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".xlsx")]

    Map_json = []
    
    for file in files:
        print(f"\nProcessing file: {file}")

        df = pd.read_excel(os.path.join(INPUT_DIR, file))
        df = clean_data(df)
        df = fill_missing_window(df)
        df = remove_outliers(df)

        if "tanggal" in df.columns:
            df.rename(columns={"tanggal": "Tanggal"}, inplace=True)

        out_path = os.path.join(OUTPUT_DIR, file)
        df.to_excel(out_path, index=False)
        print(f"Preprocessed saved: {out_path}")

        file_type = detect_file_type(df)

        station_name = os.path.splitext(file)[0]
        station, _ = Station.objects.get_or_create(
            nama_stasiun=station_name,
            defaults={'latitude': 0, 'longitude': 0}
        )

        if file_type == "polutan":
            for _, row in df.iterrows():
                insert_polutan_row(row, station)
            print(f"Insert POLUTAN OK: {station_name}")
            
            last_pol = get_last_polutan_row(df, station_name=station.nama_stasiun)
            if last_pol is not None:
                Map_json.append(last_pol)
                
            folder_path = os.path.join(BASE_DIR, "Dataset", "Website Essential")
            os.makedirs(folder_path, exist_ok=True)

            json_path = os.path.join(folder_path, "Data_Map.json")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(Map_json, f, indent=2, ensure_ascii=False)

            print(f"\nFile JSON koordinat stasiun tersimpan di: {json_path}")
            
            update_mapview_last_row(df, station)

        elif file_type == "meteo":
            for _, row in df.iterrows():
                insert_meteorologi_row(row, station)
            print(f"Insert METEOROLOGI OK: {station_name}")

        else:
            print(f"Tidak bisa deteksi tipe file: {file}")

    print("\n==== PREPROCESS_1 SELESAI ====")