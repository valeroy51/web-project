import os
import sys
import django
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

from django.conf import settings
from API.models import Station, PollutantData, MeteorologicalData

BASE_DIR = settings.BASE_DIR

INPUT_DIR = os.path.join(BASE_DIR, "Dataset", "Preprocess_1")
MERGE_DIR = os.path.join(BASE_DIR, "Dataset", "Preprocess_2", "Merged")
MINMAX_DIR = os.path.join(BASE_DIR, "Dataset", "Preprocess_2", "MinMax")

os.makedirs(MERGE_DIR, exist_ok=True)
os.makedirs(MINMAX_DIR, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def minmax_normalize(df, stasiun):
    df = df.copy()

    drop_cols = [c for c in df.columns if c.lower().startswith(("id", "station_id"))]
    df = df.drop(columns=drop_cols, errors="ignore")

    numeric_cols = [c for c in df.columns if c.lower() != "tanggal"]

    rows = []
    for col in numeric_cols:
        mn, mx = df[col].min(), df[col].max()
        rows.append({"kolom": col, "min": mn, "max": mx})
        df[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0

    pd.DataFrame(rows).to_excel(
        os.path.join(MINMAX_DIR, f"{stasiun}_MinMax.xlsx"),
        index=False
    )

    return df

def prepare_training_data():
    print("\n==============================")
    print("   PREPROCESS 2 DIMULAI")
    print("==============================\n")

    stations = Station.objects.all()

    polutan_dfs = {}
    meteo_dfs = {}

    print("\nLoad file Preprocess_1 untuk tiap stasiun...")
    for st in stations:
        file_path = os.path.join(INPUT_DIR, f"{st.nama_stasiun}.xlsx")

        if not os.path.exists(file_path):
            print(f"File tidak ditemukan: {file_path}")
            continue

        df = pd.read_excel(file_path)
        df_norm = minmax_normalize(df, st.nama_stasiun)

        if PollutantData.objects.filter(station=st).exists():
            polutan_dfs[st.id_station] = (st, df_norm)

        if MeteorologicalData.objects.filter(station=st).exists():
            meteo_dfs[st.id_station] = (st, df_norm)

    print("\nMencari pasangan stasiun berdasarkan jarak terdekat...")
    for pid, (sp, df_pol) in polutan_dfs.items():

        nearest_station = None
        min_distance = 999999

        for mid, (sm, df_met) in meteo_dfs.items():
            distance = haversine(sp.latitude, sp.longitude, sm.latitude, sm.longitude)
            if distance < min_distance:
                min_distance = distance
                nearest_station = sm

        if nearest_station is None:
            print(f"Tidak menemukan pasangan untuk {sp.nama_stasiun}")
            continue

        print(f"{sp.nama_stasiun} > {nearest_station.nama_stasiun} ({min_distance:.2f} km)")

        df_met = meteo_dfs[nearest_station.id_station][1].copy()

        df_pol = df_pol.copy()
        df_met = df_met.copy()

        for col in df_pol.columns:
            if "stasiun" in col.lower():
                df_pol = df_pol.drop(columns=[col])

        for col in df_met.columns:
            if "stasiun" in col.lower():
                df_met = df_met.drop(columns=[col])

        def normalize_tanggal(df_):
            for cand in ["tanggal", "Tanggal", "TANGGAL", "tanggal_waktu", "waktu"]:
                if cand in df_.columns:
                    df_ = df_.rename(columns={cand: "Tanggal"})
                    break
            return df_

        df_pol = normalize_tanggal(df_pol)
        df_met = normalize_tanggal(df_met)

        def parsedate(series):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s1 = pd.to_datetime(series, dayfirst=False, errors="coerce")
                s2 = pd.to_datetime(series, dayfirst=True, errors="coerce")
                return s1 if s1.notna().sum() >= s2.notna().sum() else s2

        df_pol["Tanggal"] = parsedate(df_pol["Tanggal"]).dt.date
        df_met["Tanggal"] = parsedate(df_met["Tanggal"]).dt.date

        merged = pd.merge(
            df_pol,
            df_met,
            on="Tanggal",
            how="inner",
            suffixes=("_Polu", "_Meteo")
        )

        out_name = f"{sp.nama_stasiun}_vs_{nearest_station.nama_stasiun}.xlsx"
        merged.to_excel(os.path.join(MERGE_DIR, out_name), index=False)

        print(f"Saved MERGED: {out_name}")

    print("\n==============================")
    print(" PREPROCESS 2 SELESAI")
    print("==============================\n")


if __name__ == "__main__":
    print("\n==============================")
    print(" PREPROCESS 2 DIMULAI")
    print("==============================\n")

    stations = Station.objects.all()

    polutan_dfs = {}
    meteo_dfs = {}

    print("\nLoad file Preprocess_1 untuk tiap stasiun...")
    for st in stations:
        file_path = os.path.join(INPUT_DIR, f"{st.nama_stasiun}.xlsx")

        if not os.path.exists(file_path):
            print(f"File tidak ditemukan: {file_path}")
            continue

        df = pd.read_excel(file_path)
        df_norm = minmax_normalize(df, st.nama_stasiun)

        if PollutantData.objects.filter(station=st).exists():
            polutan_dfs[st.id_station] = (st, df_norm)

        if MeteorologicalData.objects.filter(station=st).exists():
            meteo_dfs[st.id_station] = (st, df_norm)

    print("\nMencari pasangan stasiun berdasarkan jarak terdekat...")
    for pid, (sp, df_pol) in polutan_dfs.items():

        nearest_station = None
        min_distance = 999999

        for mid, (sm, df_met) in meteo_dfs.items():
            distance = haversine(sp.latitude, sp.longitude, sm.latitude, sm.longitude)
            if distance < min_distance:
                min_distance = distance
                nearest_station = sm

        if nearest_station is None:
            print(f"Tidak menemukan pasangan untuk {sp.nama_stasiun}")
            continue

        print(f"{sp.nama_stasiun} > {nearest_station.nama_stasiun} ({min_distance:.2f} km)")

        df_met = meteo_dfs[nearest_station.id_station][1].copy()

        df_pol = df_pol.copy()
        df_met = df_met.copy()

        for col in df_pol.columns:
            if "stasiun" in col.lower():
                df_pol = df_pol.drop(columns=[col])

        for col in df_met.columns:
            if "stasiun" in col.lower():
                df_met = df_met.drop(columns=[col])

        def normalize_tanggal(df_):
            for cand in ["tanggal", "Tanggal", "TANGGAL", "tanggal_waktu", "waktu"]:
                if cand in df_.columns:
                    df_ = df_.rename(columns={cand: "Tanggal"})
                    break
            return df_

        df_pol = normalize_tanggal(df_pol)
        df_met = normalize_tanggal(df_met)

        def parsedate(series):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s1 = pd.to_datetime(series, dayfirst=False, errors="coerce")
                s2 = pd.to_datetime(series, dayfirst=True, errors="coerce")
                return s1 if s1.notna().sum() >= s2.notna().sum() else s2

        df_pol["Tanggal"] = parsedate(df_pol["Tanggal"]).dt.date
        df_met["Tanggal"] = parsedate(df_met["Tanggal"]).dt.date

        merged = pd.merge(
            df_pol,
            df_met,
            on="Tanggal",
            how="inner",
            suffixes=("_Polu", "_Meteo")
        )

        out_name = f"{sp.nama_stasiun}_vs_{nearest_station.nama_stasiun}.xlsx"
        merged.to_excel(os.path.join(MERGE_DIR, out_name), index=False)

        print(f"Saved MERGED: {out_name}")
