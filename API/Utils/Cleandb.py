import os
import django
import sys
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

from API.models import (
    PollutantData, MeteorologicalData,
    ModelMSSA, PredictionResult,
    CorrelationAnalysis, Station, MapView
)

print("Membersihkan database...")

PollutantData.objects.all().delete()
MeteorologicalData.objects.all().delete()
ModelMSSA.objects.all().delete()
PredictionResult.objects.all().delete()
CorrelationAnalysis.objects.all().delete()
Station.objects.all().delete()
MapView.objects.all().delete()

print("Semua data berhasil dihapus!")
