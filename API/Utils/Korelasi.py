import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
import django
from datetime import date
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

from django.conf import settings
from API.models import CorrelationAnalysis

BASE_DIR = settings.BASE_DIR

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
        "kelembapanratarata": "rhavg",
        "curahhujan": "rr",
        "kecepatananginmaksimum": "ffx",
        "kecepatananginratarata": "ffavg",
    }

    for long_word, short in mapping.items():
        if long_word in raw:
            return short

    return cleaned[:6] or raw

nama_lengkap = {
    "tn": "Temperatur Minimum",
    "tx": "Temperatur Maksimum",
    "tavg": "Temperatur Rata Rata",
    "rhavg": "Kelembapan Rata Rata",
    "rr": "Curah Hujan",
    "ffx": "Kecepatan Angin Maksimum",
    "ffavg": "Kecepatan Angin Rata Rata",

    "pm10": "PM₁₀",
    "pm25": "PM₂.₅",
    "so2":  "SO₂",
    "co":   "CO",
    "o3":   "O₃",
    "no2":  "NO₂"
}

def calc_spearman(df, polutan_cols, meteo_cols):
    spearman = pd.DataFrame(index=polutan_cols, columns=meteo_cols, dtype=float)

    for p in polutan_cols:
        for m in meteo_cols:
            x, y = df[p], df[m]

            if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
                spearman.loc[p, m] = np.nan
                continue

            corr, _ = spearmanr(x, y, nan_policy='omit')
            spearman.loc[p, m] = corr

    return spearman

def calc_ccf(df, polutan_cols, meteo_cols, max_lag=10):

    def ccf_signed(x, y):
        x, y = np.array(x), np.array(y)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return np.nan

        x -= x.mean()
        y -= y.mean()

        vals = []

        for k in range(-max_lag, max_lag + 1):
            try:
                if k < 0:
                    corr = np.corrcoef(x[:k], y[-k:])[0, 1]
                elif k > 0:
                    corr = np.corrcoef(x[k:], y[:-k])[0, 1]
                else:
                    corr = np.corrcoef(x, y)[0, 1]
                vals.append(corr)
            except:
                vals.append(np.nan)

        vals = np.array(vals)
        return vals[np.nanargmax(np.abs(vals))] if np.any(~np.isnan(vals)) else np.nan

    ccf = pd.DataFrame(index=polutan_cols, columns=meteo_cols, dtype=float)

    for p in polutan_cols:
        for m in meteo_cols:
            ccf.loc[p, m] = ccf_signed(df[p], df[m])

    return ccf

def plot_heatmaps(spearman, ccf):
    if spearman.empty or ccf.empty:
        print("Tidak ada data valid untuk heatmap")
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(spearman, annot=True, cmap="YlGnBu", center=0, ax=ax[0])
    ax[0].set_title("Spearman Correlation")

    sns.heatmap(ccf, annot=True, cmap="RdBu_r", center=0, ax=ax[1])
    ax[1].set_title("Cross-Correlation (Signed)")

    plt.tight_layout()
    plt.show()

def classify_strength(r):
    if abs(r) >= 0.7:
        return "Kuat"
    elif abs(r) >= 0.5:
        return "Sedang"
    return "Lemah"

if __name__ == "__main__":
    print("\n=== KORELASI DIMULAI ===")

    path_file = os.path.join(BASE_DIR, "Dataset", "Preprocess_2", "Merged", "Bundaran HI_vs_Kemayoran.xlsx")
    df = pd.read_excel(path_file)

    new_cols = {}
    for c in df.columns:
        short = extract_short_code(c)
        if short in nama_lengkap:
            new_cols[c] = short
    df.rename(columns=new_cols, inplace=True)

    polutan_cols = [c for c in df.columns if c in ["pm10", "pm25", "so2", "co", "o3", "no2"]]
    meteo_cols   = [c for c in df.columns if c in ["tn", "tx", "tavg", "rhavg", "rr", "ffx", "ffavg"]]

    print("Polutan:", polutan_cols)
    print("Meteo:", meteo_cols)

    if not meteo_cols:
        print("Tidak ada kolom meteorologi valid!")
        sys.exit()

    spearman = calc_spearman(df, polutan_cols, meteo_cols)
    ccf = calc_ccf(df, polutan_cols, meteo_cols, max_lag=10)

    plot_heatmaps(spearman, ccf)

    for p in polutan_cols:
        for m in meteo_cols:
            r = spearman.loc[p, m]
            if pd.isna(r):
                continue

            CorrelationAnalysis.objects.update_or_create(
                pasangan_variabel=f"{nama_lengkap[p]} × {nama_lengkap[m]}",
                nilai_korelasi=float(r),
                tingkat_kekuatan=classify_strength(r),
                tanggal_analisis=date.today()
            )

    print("\nKorelasi berhasil disimpan ke database!")
    print("=== SELESAI ===\n")
