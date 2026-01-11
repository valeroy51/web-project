import subprocess
import sys
import os
from django.conf import settings
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

def run_script(path):
    print(f"\n==============================")
    print(f"Menjalankan: {os.path.basename(path)}")
    print("==============================")

    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print("\nERROR terjadi saat menjalankan", path)
        print(result.stderr)
        sys.exit(1)

    print(f"✔ Selesai: {os.path.basename(path)}\n")


BASE = settings.BASE_DIR

KORELASI           = os.path.join(BASE, "API", "Utils", "Korelasi.py")
BEGIN_PREPROCESS   = os.path.join(BASE, "API", "Utils", "Begin_PreProcessing.py")
PREPROCESS_1       = os.path.join(BASE, "API", "Utils", "preprocess_ke_1.py")
PREPROCESS_2       = os.path.join(BASE, "API", "Utils", "preprocess_ke_2.py")
RUNNER_MSSA        = os.path.join(BASE, "API", "Utils", "Runner_MSSA.py")


if __name__ == "__main__":

    print("\n====================================")
    print("    PIPELINE OTOMATIS DIMULAI ")
    print("====================================\n")

    run_script(PREPROCESS_1)
    run_script(BEGIN_PREPROCESS)
    run_script(PREPROCESS_2)
    run_script(KORELASI)
    run_script(RUNNER_MSSA)

    print("\n====================================")
    print("   SEMUA PROSES SELESAI TANPA ERROR! ")
    print("====================================\n")