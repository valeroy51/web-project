import os
import sys
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

import glob
from django.conf import settings

from API.Utils.Program_MSSA import run_mssa_pipeline

def run_mssa_for_all():
    BASE = settings.BASE_DIR
    
    MERGE_DIR = os.path.join(BASE, "Dataset", "Preprocess_2", "Merged")
    NORM_DIR  = os.path.join(BASE, "Dataset", "Preprocess_2", "MinMax")
    OUT_ROOT  = os.path.join(BASE, "Dataset", "MSSA")

    os.makedirs(OUT_ROOT, exist_ok=True)

    merged_files = glob.glob(os.path.join(MERGE_DIR, "*.xlsx"))
    if not merged_files:
        return ["Tidak ada file hasil merge di folder Merged/"]

    hasil_semua = []

    for path in merged_files:
        nama_file = os.path.splitext(os.path.basename(path))[0]

        if "_vs_" not in nama_file:
            continue

        pol, met = nama_file.split("_vs_", 1)

        norm_pol = os.path.join(NORM_DIR, f"{pol}_MinMax.xlsx")
        norm_met = os.path.join(NORM_DIR, f"{met}_MinMax.xlsx")

        out_dir = os.path.join(OUT_ROOT, nama_file)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[MSSA] Mulai: {nama_file}")

        hasil = run_mssa_pipeline(
            DATA_PATH=path,
            NORM_POL_PATH=norm_pol,
            NORM_MET_PATH=norm_met,
            OUT_DIR=out_dir,
            energy_thr=0.97,
            l_min=40,
            l_max=40,
            n_jobs_inner=-1
        )

        hasil_semua.append({
            "pair": nama_file,
            "result": hasil
        })
        
        if hasil_semua:
            import pandas as pd
            df_summary = pd.DataFrame([h["result"] for h in hasil_semua])
            summary_path = os.path.join(OUT_ROOT, "Master_Summary.xlsx")
            df_summary.to_excel(summary_path, index=False)
            print(f"[MSSA] Master summary tersimpan di: {summary_path}")

        print(f"[MSSA] Selesai: {nama_file}")

    return hasil_semua

if __name__ == "__main__":
    run_mssa_for_all()