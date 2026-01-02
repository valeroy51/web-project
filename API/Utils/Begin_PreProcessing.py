import os
import glob
import pandas as pd
import requests
import time
import django
import sys
from pathlib import Path
import django

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Skripsi.settings")
django.setup()

from django.conf import settings
BASE_DIR = settings.BASE_DIR

from API.models import Station

def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "jsonv2", "limit": 1, "polygon_geojson":1}
    headers = {"referer": "https://nominatim.openstreetmap.org/ui/search.html?q={address}"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return {"alamat": address, "lat": lat, "lon": lon}
    return {"alamat": address, "lat": None, "lon": None}

def load_excel_names(folder_path):
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    df_names = []

    for file in excel_files:
        df_name = os.path.splitext(os.path.basename(file))[0]
        df_names.append(df_name)
        print(f"Ditemukan file: {df_name}.xlsx")

    return df_names

def save_stations_to_db(folder_path):
    print("\nMemulai proses simpan lokasi stasiun ke database...\n")

    df_names = load_excel_names(folder_path)
    time.sleep(1)

    hasil_geojson = []
    
    for name in df_names:
        loc_name = name.replace("_", " ").strip()
        address = f"{loc_name}, Indonesia"

        geo = geocode_address(address)
        print(address)

        print(f"{name} > Lat: {geo['lat']}, Lon: {geo['lon']}")

        if geo["lat"] is None or geo["lon"] is None:
            print(f"Koordinat tidak ditemukan — SKIP {name}")
            time.sleep(1)
            continue

        Station.objects.update_or_create(
            nama_stasiun=name,
            defaults={'latitude': geo['lat'], 'longitude': geo['lon']}
        )

        hasil_geojson.append({
            "nama_stasiun": name,
            "latitude": geo["lat"],
            "longitude": geo["lon"]
        })
        
        time.sleep(1)

    folder_path = os.path.join(BASE_DIR, "Dataset", "Website Essential")
    os.makedirs(folder_path, exist_ok=True)
    
    json_path = os.path.join(folder_path, "Stasiun_coordinate.json")

    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(hasil_geojson, f, indent=2, ensure_ascii=False)

    print(f"\nFile JSON koordinat stasiun tersimpan di: {json_path}")
        
    print("\nSemua titik stasiun sudah disimpan ke database Django!")

if __name__ == "__main__":
    folder_path = os.path.join(BASE_DIR, "Dataset", "Preprocess_1")
    save_stations_to_db(folder_path)