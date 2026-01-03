import os
import sys
import json
import logging
import warnings
from babel.dates import format_date
from django.http import JsonResponse
import difflib
from collections import Counter
from datetime import date, timedelta
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import ObjectDoesNotExist
from django.contrib import messages
from django.db.models import Max
from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.contrib.auth import logout
from django.contrib.auth.decorators import user_passes_test
from django.http import JsonResponse
from django.shortcuts import render
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404
from API.Utils.preprocess_ke_1 import preprocess_uploaded_df
from API.models import Station
from API.models import PollutantData
from API.models import MeteorologicalData
from API.models import PredictionResult
from API.models import CorrelationAnalysis
from API.models import MapView
from API.models import ModelMSSA
from API.forms import PollutantDataForm
from API.forms import MeteorologicalDataForm
import json, os
from django.shortcuts import redirect
from django.contrib import messages
from django.utils import timezone
from datetime import datetime, timedelta, timezone
import json, os
import threading
from functools import wraps


logger = logging.getLogger(__name__)

format_date(datetime.now(), "EEEE, d MMMM y", locale="id")

SCHEDULE_FILE = "training_schedule.json"

def admin_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        masked_pw = "*" * len(password) if password else ""
        logger.warning(f"[LOGIN ATTEMPT] Username: {username} | Password Masked: {masked_pw}")

        user = authenticate(request, username=username, password=password)

        if user is None:
            logger.warning(f"[LOGIN FAILED] Username '{username}' gagal login.")

            if wants_json(request):
                return JsonResponse(
                    {"status": "error", "message": "Username atau password salah"},
                    status=401
                )

            messages.error(request, "Username atau password salah.")
            return redirect("admin_login")

        if not user.is_staff:
            logger.warning(f"[LOGIN BLOCKED] User '{username}' bukan admin.")

            if wants_json(request):
                return JsonResponse(
                    {"status": "forbidden", "message": "Akun ini bukan admin"},
                    status=403
                )

            messages.error(request, "Akses ditolak. Akun ini bukan admin.")
            return redirect("admin_login")

        # sukses
        login(request, user)
        logger.warning(f"[LOGIN SUCCESS] User '{username}' berhasil login sebagai admin.")

        if wants_json(request):
            return JsonResponse({
                "status": "ok",
                "message": "Login berhasil",
                "redirect": "/",  # home
            })

        return redirect("home")

    # GET
    if wants_json(request):
        return JsonResponse(
            {"detail": "Gunakan method POST"},
            status=405
        )

    return render(request, "admin_login.html")

def admin_logout(request):
    logout(request)

    if wants_json(request):
        return JsonResponse({
            "status": "ok",
            "message": "Logout berhasil",
            "redirect": "/",  # home
        })

    return redirect("home")

def wants_json(request):
    accept = (request.headers.get("Accept") or "").lower()
    return (
        request.GET.get("format") == "json"
        or "application/json" in accept
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
    )

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            if wants_json(request):
                return JsonResponse(
                    {"status": "unauthorized", "message": "Silakan login terlebih dahulu"},
                    status=401
                )
            messages.error(request, "Silakan login terlebih dahulu.")
            return redirect("admin_login")

        if not request.user.is_staff:
            if wants_json(request):
                return JsonResponse(
                    {"status": "forbidden", "message": "Akses ditolak. Akun ini bukan admin"},
                    status=403
                )
            messages.error(request, "Akses ditolak. Akun ini bukan admin.")
            return redirect("home")

        return view_func(request, *args, **kwargs)
    return wrapper

def polutan_info(request):
    if wants_json(request):
        return JsonResponse({"status": "ok", "page": "penjelasan"})
    return render(request, "penjelasan.html")

def home(request):
    stations = Station.objects.exclude(
        nama_stasiun__icontains="Halim"
    ).exclude(
        nama_stasiun__icontains="Kemayoran"
    ).exclude(
        nama_stasiun__icontains="Maritim"
    )

    station_data = []

    def kategori_ispu(nilai):
        if nilai <= 50:
            return "Baik"
        elif nilai <= 100:
            return "Sedang"
        elif nilai <= 200:
            return "Tidak Sehat"
        elif nilai <= 300:
            return "Sangat Tidak Sehat"
        return "Berbahaya"

    color_map = {
        "Baik": "green",
        "Sedang": "blue",
        "Tidak Sehat": "orange",
        "Sangat Tidak Sehat": "red",
        "Berbahaya": "black",
    }

    for s in stations:
        mv = MapView.objects.filter(station=s).order_by("-tanggal").first()

        if mv:
            pm_values = [mv.pm10, mv.pm25, mv.so2, mv.co, mv.o3, mv.no2]
            max_ispu = max([v or 0 for v in pm_values])
            status = kategori_ispu(max_ispu)
            color = color_map.get(status, "gray")

            polutan_mv= {
                    "PM<sub>10</sub>": mv.pm10 or 0,
                    "PM<sub>2.5</sub>": mv.pm25 or 0,
                    "SO<sub>2</sub>": mv.so2 or 0,
                    "CO": mv.co or 0,
                    "O<sub>3</sub>": mv.o3 or 0,
                    "NO<sub>2</sub>": mv.no2 or 0,
                }
            
            polutan_tertinggi = max(polutan_mv, key=polutan_mv.get)
            nilai_tertinggi = polutan_mv[polutan_tertinggi]
            
            station_data.append({
                "id": s.id_station,
                "name": s.nama_stasiun.replace(", Indonesia", "").strip(),
                "coords": [s.latitude, s.longitude],
                "status": status,
                "status_clean": status,
                "color": color,
                "tanggal": mv.tanggal.strftime("%d %B %Y"),
                "ispu": round(nilai_tertinggi, 1),
                "polutan_utama": polutan_tertinggi,
                "data": {k: round(v, 1) for k, v in polutan_mv.items()},
            })

        else:
            pred = PredictionResult.objects.filter(station=s).order_by("tanggal_prediksi").first()
            if pred:
                pm_values = [pred.pm10_pred, pred.pm25_pred, pred.so2_pred, pred.co_pred, pred.o3_pred, pred.no2_pred]
                max_ispu = max([v or 0 for v in pm_values])
                status = kategori_ispu(max_ispu)
                color = color_map.get(status, "gray")

                polutan_pred = {
                    "PM<sub>10</sub>": pred.pm10_pred or 0,
                    "PM<sub>2.5</sub>": pred.pm25_pred or 0,
                    "SO<sub>2</sub>": pred.so2_pred or 0,
                    "CO": pred.co_pred or 0,
                    "O<sub>3</sub>": pred.o3_pred or 0,
                    "NO<sub>2</sub>": pred.no2_pred or 0,
                }
                polutan_tertinggi = max(polutan_pred, key=polutan_pred.get)
                nilai_tertinggi = polutan_pred[polutan_tertinggi]

                station_data.append({
                    "id": s.id_station,
                    "name": s.nama_stasiun.replace(", Indonesia", "").strip(),
                    "coords": [s.latitude, s.longitude],
                    "status": status,
                    "status_clean": status,
                    "color": color,
                    "tanggal": "-",
                    "ispu": round(nilai_tertinggi, 1),
                    "polutan_utama": polutan_tertinggi,
                    "data": {k: round(v, 1) for k, v in polutan_pred.items()},
                })
            else:
                station_data.append({
                    "id": s.id_station,
                    "name": s.nama_stasiun.replace(", Indonesia", "").strip(),
                    "coords": [s.latitude, s.longitude],
                    "status": "Tidak Ada Data",
                    "status_clean": "Tidak Ada Data",
                    "color": "gray",
                    "tanggal": " - ",
                    "ispu": 0,
                    "polutan_utama": "",
                    "data": {"PM<sub>10</sub>": "-", "PM<sub>2.5</sub>": "-", "SO<sub>2</sub>": "-", "CO": "-", "O<sub>3</sub>": "-", "NO<sub>2</sub>": "-"},
                })
    realtime = {
        "Baik": 0,
        "Sedang": 0,
        "Tidak_Sehat": 0,
        "Sangat_Tidak_Sehat": 0,
        "Berbahaya": 0,
    }

    for data in station_data:
        status = data["status_clean"]
        if status in realtime:
            realtime[status] += 1

    if station_data:
        sorted_stations = sorted(station_data, key=lambda x: x["ispu"])

        best_station = sorted_stations[0]["name"]
        worst_station = sorted_stations[-1]["name"]
    else:
        best_station = "-"
        worst_station = "-"

    # return render(request, "home.html", {"stations": json.dumps(station_data), "realtime": realtime, "best_station": best_station, "worst_station": worst_station})

    payload = {
        "stations": station_data,
        "realtime": realtime,
        "best_station": best_station,
        "worst_station": worst_station,
    }

    if wants_json(request):
        return JsonResponse(payload)

    payload["stations_json"] = json.dumps(station_data)
    return render(request, "home.html", payload)

@admin_required
def data(request):
    def safe_float(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            if value in ["-", "–", "", "N/A", "na", "NaN"]:
                return None
        try:
            val = float(value)
            if abs(val) > 1000:
                return None
            return val
        except Exception:
            return None

    if not cache.get('server_alive'):
        cache.set('server_alive', True, timeout=None)
        request.session['upload_history'] = []
        request.session.modified = True
        print("Server baru dimulai — upload_history dibersihkan otomatis.")


    polutan_form = PollutantDataForm()
    meteo_form = MeteorologicalDataForm()
    uploaded_filename = None
    file_type = None
    active_tab = None

    station_names = list(Station.objects.values_list('nama_stasiun', flat=True))
    station_names_lower = [s.lower() for s in station_names]

    max_allowed_date = date.today() + timedelta(days=1)
    min_allowed_date = date(2022, 1, 1)

    def parsedate(series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            numeric = pd.to_numeric(series, errors="coerce")
            converted = series.copy()

            mask = numeric.notna() & (numeric > 1000)
            converted.loc[mask] = pd.to_datetime(
                numeric[mask], unit="D", origin="1899-12-30", errors="coerce"
            )

            s1 = pd.to_datetime(converted, dayfirst=False, errors="coerce")
            s2 = pd.to_datetime(converted, dayfirst=True, errors="coerce")
            best = s1 if s1.notna().sum() >= s2.notna().sum() else s2
            return best.dt.date

    def find_closest_station(name):
        if not isinstance(name, str) or not name.strip():
            return None

        name = name.strip().lower().replace(",", " ").replace(".", " ")
        name = " ".join(name.split()) 

        for s in station_names_lower:
            if name in s or s in name:
                idx = station_names_lower.index(s)
                return station_names[idx]

        match = difflib.get_close_matches(name, station_names_lower, n=1, cutoff=0.45)
        if match:
            idx = station_names_lower.index(match[0])
            return station_names[idx]

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
            "curahhujan": "rr",
            "kecepatananginmaksimum": "ffx",
            "kecepatananginratarata": "ffavg",
        }

        for long_word, short_code in mapping.items():
            if long_word in raw:
                return short_code

        return cleaned[:5] or raw

    if request.method == "POST":
        if 'save_pollutant' in request.POST:
            polutan_form = PollutantDataForm(request.POST)
            if polutan_form.is_valid():
                tanggal_input = polutan_form.cleaned_data.get('tanggal')
                station_input = polutan_form.cleaned_data.get('station')

                if tanggal_input > max_allowed_date:
                    messages.error(request, f"Tanggal {tanggal_input} melebihi {max_allowed_date}.")
                elif tanggal_input < min_allowed_date:
                    messages.error(request, f"Tanggal {tanggal_input} sebelum {min_allowed_date}.")
                else:
                    obj, created = PollutantData.objects.update_or_create(
                        station=station_input,
                        tanggal=tanggal_input,
                        defaults=polutan_form.cleaned_data
                    )
                    if created:
                        messages.success(request, "Data polutan baru disimpan.")
                    else:
                        messages.info(request, "Data polutan lama diperbarui.")
                # return redirect('/input/')
                
                if wants_json(request):
                    return JsonResponse({
                        "status": "ok",
                        "action": "save_pollutant",
                        "created": created,
                        "message": "Data polutan baru disimpan." if created else "Data polutan lama diperbarui.",
                    })
                return redirect("/input/")


        elif 'save_meteo' in request.POST:
            meteo_form = MeteorologicalDataForm(request.POST)
            if meteo_form.is_valid():
                tanggal_input = meteo_form.cleaned_data.get('tanggal')
                station_input = meteo_form.cleaned_data.get('station')

                if tanggal_input > max_allowed_date:
                    messages.error(request, f"Tanggal {tanggal_input} melebihi {max_allowed_date}.")
                elif tanggal_input < min_allowed_date:
                    messages.error(request, f"Tanggal {tanggal_input} sebelum {min_allowed_date}.")
                else:
                    obj, created = MeteorologicalData.objects.update_or_create(
                        station=station_input,
                        tanggal=tanggal_input,
                        defaults=meteo_form.cleaned_data
                    )
                    if created:
                        messages.success(request, f"Data meteorologi baru disimpan untuk {station_input.nama_stasiun} ({tanggal_input}).")
                    else:
                        messages.info(request, f"Data meteorologi untuk {station_input.nama_stasiun} tanggal {tanggal_input} telah diperbarui.")
                # return redirect('/input/')
                
                if wants_json(request):
                    return JsonResponse({
                        "status": "ok",
                        "action": "save_meteo",
                        "created": created,
                        "message": "Data meteorologi baru disimpan." if created else "Data meteorologi lama diperbarui.",
                    })
                return redirect("/input/")

        elif 'upload' in request.POST and request.FILES.get('upload_file'):
            file = request.FILES['upload_file']
            uploaded_filename = file.name
            ext = file.name.split('.')[-1].lower()

            try:
                if ext == 'csv':
                    df = pd.read_csv(file)
                elif ext in ['xlsx', 'xls']:
                    df = pd.read_excel(file)
                else:
                    messages.error(request, "Format file tidak didukung.")
                    
                    if wants_json(request):
                        return JsonResponse({"status": "error", "message": "Format file tidak didukung."}, status=400)
                    return redirect("input")
                
            except Exception as e:
                messages.error(request, f"Gagal membaca file: {e}")
                if wants_json(request):
                    return JsonResponse({"status": "error", "message": "Gagal membaca file."}, status=400)
                return redirect("input")
            
            if df.empty or df.dropna(how="all").shape[0] == 0:
                messages.error(request, "File kosong! Pastikan Anda sudah mengisi data sebelum mengunggah template.")
                if wants_json(request):
                    return JsonResponse({"status": "error", "message": "File kosong!."}, status=400)
                return redirect("input")

            if 'tanggal' not in [c.lower().strip().replace(' ', '') for c in df.columns]:
                messages.error(request, "File tidak memiliki kolom 'tanggal'. Pastikan format template sesuai.")
                if wants_json(request):
                    return JsonResponse({"status": "error", "message": "File tidak memiliki kolom 'tanggal'."}, status=400)
                return redirect("input")


            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(' ', '', regex=False)
                .str.replace('_', '', regex=False)
                .str.replace('.', '', regex=False)
            )

            pollutant_cols = {'pm10', 'pm25', 'so2', 'co', 'o3', 'no2'}
            meteo_cols = {'tn', 'tx', 'tavg', 'rhavg', 'rr', 'ffx', 'ffavg'}
            

            if 'tanggal' not in df.columns:
                messages.error(request, "Kolom 'tanggal' tidak ditemukan di file.")
                if wants_json(request):
                    return JsonResponse({"status": "error", "message": "Kolom 'tanggal' tidak ditemukan di file."}, status=400)
                return redirect("data")
            
            df['tanggal'] = parsedate(df['tanggal'])
            df = df[df['tanggal'].notna()]
            df = df[(df['tanggal'] >= min_allowed_date) & (df['tanggal'] <= max_allowed_date)]

            print("\n=== DEBUG TANGGAL ===")
            print(f"Total data valid tanggal: {len(df)}")
            if not df.empty:
                print(f"Rentang tanggal: {df['tanggal'].min()} -> {df['tanggal'].max()}")
                print(f"Contoh tanggal: {df['tanggal'].head(3).to_list()}")

            if df.empty:
                messages.error(request, "Tidak ada data dalam rentang tanggal valid.")
                if wants_json(request):
                    return JsonResponse({"status": "error", "message": "Tidak ada data dalam rentang tanggal valid."}, status=400)
                return redirect("data")

            inserted, skipped = 0, 0
            matched_log = []

            print("\n=== DEBUG STASIUN ===")
            new_cols = []
            for col in df.columns:
                code = extract_short_code(col)
                if code in meteo_cols:
                    new_cols.append(code)
                else:
                    new_cols.append(col)
            df.columns = new_cols
            
            if pollutant_cols.intersection(df.columns):
                file_type = "polutan"
                
                print("\n=== PREPROCESSING DIMULAI ===")
                df_clean = preprocess_uploaded_df(df.copy(), station=None, file_type="polutan")


                OUTPUT_PRE = os.path.join(settings.BASE_DIR, "Dataset", "Preprocessed Website")
                os.makedirs(OUTPUT_PRE, exist_ok=True)

                pre_name = f"preprocessed_{uploaded_filename}"
                pre_path = os.path.join(OUTPUT_PRE, pre_name)

                base, ext = os.path.splitext(pre_path)
                ext = ext.lower()

                if ext in [".csv", ".xls", ".xlsx"]:
                    pre_path = base + ".xlsx"

                df_clean.to_excel(pre_path, index=False)

                print(f"Preprocessing disimpan ke: {pre_path}")
                print("=== PREPROCESSING SELESAI ===\n")
                df = df_clean
                
                for idnumber, row in df.iterrows():
                    station_col = None
                    for c in df.columns:
                        if 'station' in c or 'stasiun' in c or 'lokasi' in c:
                            station_col = c
                            break

                    if not station_col:
                        print("Tidak ada kolom stasiun di file, kolom saat ini:", df.columns.to_list())
                        messages.error(request, "Tidak ada kolom stasiun di file!")
                        if wants_json(request):
                            return JsonResponse({"status": "error", "message": "Tidak ada kolom stasiun di file!."}, status=400)
                        return redirect("data")

                    stasiun_input = str(row.get(station_col, '')).strip().lower()
                    guessed_name = find_closest_station(stasiun_input)

                    if guessed_name:
                        print(f"Input: Baris {idnumber} di {stasiun_input}  ->  Match: {guessed_name}")
                        station = Station.objects.filter(nama_stasiun=guessed_name).first()
                        if not station:
                            skipped += 1
                            continue
                        PollutantData.objects.update_or_create(
                            station=station,
                            tanggal=row['tanggal'],
                            defaults={
                                'pm10': row.get('pm10'),
                                'pm25': row.get('pm25'),
                                'so2': row.get('so2'),
                                'co': row.get('co'),
                                'o3': row.get('o3'),
                                'no2': row.get('no2'),
                            }
                        )
                        inserted += 1
                    else:
                        print(f"Input: Baris {idnumber} di {stasiun_input}  ->  Tidak ditemukan")
                        skipped += 1

            elif meteo_cols.intersection(df.columns):
                file_type = "meteorologi"
                
                print("\n=== PREPROCESSING DIMULAI ===")
                df_clean = preprocess_uploaded_df(df.copy(), station=None, file_type=None)

                OUTPUT_PRE = os.path.join(settings.BASE_DIR, "Dataset", "Preprocessed Website")
                os.makedirs(OUTPUT_PRE, exist_ok=True)

                pre_name = f"preprocessed_{uploaded_filename}"
                pre_path = os.path.join(OUTPUT_PRE, pre_name)

                base, ext = os.path.splitext(pre_path)
                ext = ext.lower()

                if ext in [".csv", ".xls", ".xlsx"]:
                    pre_path = base + ".xlsx"

                df_clean.to_excel(pre_path, index=False)

                print(f"Preprocessing disimpan ke: {pre_path}")
                print("=== PREPROCESSING SELESAI ===\n")

                df = df_clean
                
                for idnumber, row in df.iterrows():
                    station_col = None
                    for c in df.columns:
                        if 'station' in c or 'stasiun' in c or 'lokasi' in c:
                            station_col = c
                            break

                    if not station_col:
                        print("Tidak ada kolom stasiun di file, kolom saat ini:", df.columns.to_list())
                        messages.error(request, "Tidak ada kolom stasiun di file!")
                        if wants_json(request):
                            return JsonResponse({"status": "error", "message": "Tidak ada kolom stasiun di file!."}, status=400)
                        return redirect("data")

                    stasiun_input = str(row.get(station_col, '')).strip().lower()
                    

                    guessed_name = find_closest_station(stasiun_input)

                    if guessed_name:
                        print(f"Input: Baris {idnumber} di {stasiun_input}  ->  Match: {guessed_name}")
                        station = Station.objects.filter(nama_stasiun=guessed_name).first()
                        
                        if not station:
                            skipped += 1
                            continue
                        MeteorologicalData.objects.update_or_create(
                            station=station,
                            tanggal=row['tanggal'],
                            defaults={
                                'temperatur_minimum': safe_float(row.get('tn')), 
                                'temperatur_maksimum': safe_float(row.get('tx')), 
                                'temperatur_rata': safe_float(row.get('tavg')), 
                                'kelembapan_rata': safe_float(row.get('rhavg')), 
                                'curah_hujan': safe_float(row.get('rr')), 
                                'kecepatan_angin_maksimum': safe_float(row.get('ffx')), 
                                'kecepatan_angin_rata': safe_float(row.get('ffavg')),
                            }
                        )
                        inserted += 1
                    else:
                        print(f"Input: Baris {idnumber} di {stasiun_input}  ->  Tidak ditemukan")
                        skipped += 1

            else:
                file_type = "tidak_dikenali"
                messages.warning(request, f"Kolom file tidak dikenali: {list(df.columns)}")
                print(list(df.columns))

            print(f"\nRekap: {inserted} baris dikenali, {skipped} gagal match.\n")

            messages.success(request, f"{inserted} baris berhasil diunggah ({file_type}).")
            if skipped:
                messages.warning(request, f"{skipped} baris dilewati (stasiun tidak dikenali atau tidak valid).")

            active_tab = "upload"
            
            if wants_json(request):
                return JsonResponse({
                    "status": "ok",
                    "action": "upload",
                    "file_name": uploaded_filename,
                    "file_type": file_type,
                    "inserted": inserted,
                    "skipped": skipped,
                    "upload_history": request.session.get("upload_history", []),
                })

            
        if uploaded_filename and inserted > 0:
            upload_history = request.session.get('upload_history', [])

            new_entry = {
                'name': uploaded_filename,
                'type': file_type,
                'inserted': inserted,
                'skipped': skipped,
            }

            upload_history.insert(0, new_entry)

            upload_history = upload_history[:5]

            request.session['upload_history'] = upload_history
            request.session.modified = True
        else:
            print("Upload tidak ditambahkan ke riwayat karena tidak ada data yang berhasil dimasukkan.")

    # context = {
    #     'polutan_form': polutan_form,
    #     'meteo_form': meteo_form,
    #     'uploaded_filename': uploaded_filename,
    #     'file_type': file_type,
    #     'active_tab': active_tab,
    #     'upload_history': request.session.get('upload_history', []),
    # }
    # return render(request, 'data.html', context)
    
    context = {
        "polutan_form": polutan_form,
        "meteo_form": meteo_form,
        "uploaded_filename": uploaded_filename,
        "file_type": file_type,
        "active_tab": active_tab,
        "upload_history": request.session.get("upload_history", []),
    }

    if wants_json(request):
        return JsonResponse({
            "status": "ok",
            "upload_history": context["upload_history"],
            "active_tab": context["active_tab"],
            "uploaded_filename": context["uploaded_filename"],
            "file_type": context["file_type"],
        })

    return render(request, "data.html", context)

@admin_required
def merge(request):
    if request.method != "POST":
        if wants_json(request):
            return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)
        messages.error(request, "Method tidak diizinkan.")
        return redirect("input")

    from API.Utils.preprocess_ke_2 import prepare_training_data

    def job():
        try:
            prepare_training_data()
        except Exception as e:
            print("[MERGE ERROR]", e)

    threading.Thread(target=job, daemon=True).start()

    if wants_json(request):
        return JsonResponse({
            "status": "started",
            "message": "Merging data berjalan di background",
        })

    messages.success(request, "Merging data sedang diproses.")
    return redirect("input")

@admin_required
def train(request):
    if request.method != "POST":
        if wants_json(request):
            return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)
        messages.error(request, "Method tidak diizinkan.")
        return redirect("input")
        
    from API.Utils.Runner_MSSA import run_mssa_for_all
    
    def job():
        try:
            run_mssa_for_all()
        except Exception as e:
            print("[MERGE ERROR]", e)

    threading.Thread(target=job, daemon=True).start()

    if wants_json(request):
        return JsonResponse({
            "status": "started",
            "message": "Training MSSA sedang berjalan di background",
        })

    messages.success(request, "Training MSSA sedang berjalan.")
    return redirect("input")

@admin_required
def schedule_train(request):
    if request.method != "POST":
        if wants_json(request):
            return JsonResponse({"status": "error", "message": "Method not allowed"}, status=405)
        messages.error(request, "Method tidak diizinkan.")
        return redirect("input")
    
    if request.method == "POST":
        schedule_time = request.POST.get("schedule_time")
        offset_minutes = int(request.POST.get("tz_offset"))  # contoh: -420

        dt_naive = datetime.fromisoformat(schedule_time)

        user_tz = timezone(timedelta(minutes=-offset_minutes))

        dt = dt_naive.replace(tzinfo=user_tz)

        with open(SCHEDULE_FILE, "w") as f:
            json.dump({"scheduled_at": dt.isoformat()}, f)

        if wants_json(request):
            return JsonResponse({
                "status": "ok",
                "scheduled_at": dt.isoformat(),
                "message": "Training berhasil dijadwalkan"
            })

        messages.success(request, f"Training dijadwalkan pada {dt}")
        return redirect("input")
        # return JsonResponse({"status":"ok", "scheduled_at": dt.isoformat()})
    
def tentang(request):
    team = [
        {"name": "Valeroy Putra Sientika", "role": "Mahasiswa"},
        {"name": "Lely Hiryanto ST., M.Sc.,PH.D", "role": "Dosen Pembimbing 1"},
        {"name": "Janson Hendryli S.Kom. M.KOM", "role": "Dosen Pembimbing 2"},
    ]
    # return render(request, "tentang.html", {"team": team})

    payload = {
        "team": team,
    }

    if wants_json(request):
        return JsonResponse(payload, safe=False)

    return render(request, "tentang.html", payload)

def analisis(request):
    korelasi = CorrelationAnalysis.objects.all().order_by("-tanggal_analisis")

    polutans = set()
    meteos = set()
    korelasi_dict = {}

    for row in korelasi:
        try:
            p1, p2 = [s.strip() for s in row.pasangan_variabel.split("×")]
        except:
            continue

        if "PM" in p1 or "SO₂" in p1 or "PM₂" in p1 or "NO₂" in p1 or "O₃" in p1 or "CO" in p1:
            pol = p1
            met = p2
        else:
            pol = p2
            met = p1

        polutans.add(pol)
        meteos.add(met)

        korelasi_dict[(pol, met)] = float(row.nilai_korelasi)

    polutans = sorted(polutans)
    meteos = sorted(meteos)

    heatmap_data = []
    for pol in polutans:
        for met in meteos:
            heatmap_data.append({
                "x": met,
                "y": pol,
                "v": korelasi_dict.get((pol, met), 0.0)
            })

    if korelasi:
        best = max(korelasi, key=lambda r: float(r.nilai_korelasi))
        top_title = best.pasangan_variabel.replace("×", " - ")
        top_value = round(float(best.nilai_korelasi), 2)
    else:
        top_title, top_value = "-", 0.0

    fields_pred   = ['pm10_pred', 'pm25_pred', 'so2_pred', 'no2_pred', 'o3_pred', 'co_pred']
    labels_pol    = ['PM₁₀', 'PM₂.₅', 'SO₂', 'CO', 'O₃', 'NO₂']

    latest_pred_per_station = (
        PredictionResult.objects.values("station_id")
        .annotate(max_date=Max("tanggal_prediksi"))
    )

    latest_pred_qs = PredictionResult.objects.filter(
        station_id__in=[r["station_id"] for r in latest_pred_per_station],
        tanggal_prediksi__in=[r["max_date"] for r in latest_pred_per_station]
    ).select_related("station")

    avg_pred_values = []
    for f in fields_pred:
        vals = [getattr(p, f) for p in latest_pred_qs if getattr(p, f) is not None]
        avg_pred_values.append(round(float(np.mean(vals)), 2) if vals else 0.0)

    ispu_station_data = {}
    for p in latest_pred_qs:
        row = []
        for f in fields_pred:
            v = getattr(p, f)
            row.append(round(float(v), 2) if v else 0.0)
        ispu_station_data[p.station.nama_stasiun] = row

    polutan_dict = dict(zip(labels_pol, avg_pred_values))
    top_pol_pred = max(polutan_dict, key=polutan_dict.get) if polutan_dict else "-"
    top_val_pred = polutan_dict.get(top_pol_pred, 0.0)

    fields_mv   = ['pm10', 'pm25', 'so2', 'no2', 'o3', 'co']
    bar_labels  = ['PM₁₀', 'PM₂.₅', 'SO₂', 'CO', 'O₃', 'NO₂']

    latest_mv_per_station = (
        MapView.objects.values("station_id")
        .annotate(max_date=Max("tanggal"))
    )

    latest_mv_qs = MapView.objects.filter(
        station_id__in=[r["station_id"] for r in latest_mv_per_station],
        tanggal__in=[r["max_date"] for r in latest_mv_per_station]
    )

    bar_actual_ispu = []
    for f in fields_mv:
        vals = [getattr(m, f) for m in latest_mv_qs if getattr(m, f) is not None]
        bar_actual_ispu.append(round(float(np.mean(vals)), 2) if vals else 0.0)

    avg_ispu = round(float(np.mean(bar_actual_ispu)), 1) if bar_actual_ispu else 0.0

    if avg_pred_values and bar_actual_ispu and bar_actual_ispu[1] > 0:
        diff_pm25 = avg_pred_values[1] - bar_actual_ispu[1]
        percent_diff = round((diff_pm25 / bar_actual_ispu[1]) * 100, 1)
    else:
        percent_diff = 0.0

    cards = [
        {"icon": "🌡️", "title": f"{top_title}", "desc": "Korelasi terkuat antar parameter", "value": f"{top_value:+.2f}"},
        {"icon": "💨", "title": "Polutan Dominan", "desc": f"Kadar {top_pol_pred} tertinggi dari hasil prediksi", "value": f"{top_val_pred}"},
        {"icon": "🔁", "title": "Prediksi vs Aktual (PM₂.₅)", "desc": "Selisih rata-rata PM₂.₅ terhadap data aktual", "value": f"{percent_diff:+.1f}%"},
        {"icon": "🌧️", "title": "Rata-rata ISPU DKI", "desc": "Rata-rata kualitas udara seluruh stasiun", "value": f"{avg_ispu}"},
    ]

    # return render(request, "analisis.html", {
    #     "cards": cards,
    #     "heatmap_data": heatmap_data,
    #     "ispu_labels": labels_pol,
    #     "ispu_values_avg": avg_pred_values,
    #     "ispu_station_data": ispu_station_data,
    #     "bar_labels": bar_labels,
    #     "bar_actual_ispu": bar_actual_ispu,
    # })
    
    payload = {
        "cards": cards,
        "heatmap_data": heatmap_data,
        "ispu_labels": labels_pol,
        "ispu_values_avg": avg_pred_values,
        "ispu_station_data": ispu_station_data,
        "bar_labels": bar_labels,
        "bar_actual_ispu": bar_actual_ispu,
    }

    if wants_json(request):
        return JsonResponse(payload, safe=False)

    return render(request, "analisis.html", payload)

def prediksi(request, id_station):
    stasiun = get_object_or_404(Station, id_station=id_station)

    data_prediksi = (
        PredictionResult.objects.filter(station=stasiun)
        .order_by("-tanggal_prediksi")[:7]
    )

    data7hari = []
    for p in reversed(data_prediksi): 
        day_name = p.tanggal_prediksi.strftime("%d %b %y")
        data7hari.append({
            "day": day_name,
            "pm10": round(p.pm10_pred, 1),
            "pm25": round(p.pm25_pred, 1),
            "so2": round(p.so2_pred, 1),
            "co": round(p.co_pred, 1),
            "o3": round(p.o3_pred, 1),
            "no2": round(p.no2_pred, 1),
            "status": p.indeks_kualitas_udara.split("(")[0].strip()
        })

    mv = MapView.objects.filter(station=stasiun).order_by("-tanggal").first()

    if mv:
        today_data = {
            "day": mv.tanggal.strftime("%d %b %y"),
            "pm10": round(mv.pm10 or 0, 1),
            "pm25": round(mv.pm25 or 0, 1),
            "so2": round(mv.so2 or 0, 1),
            "co": round(mv.co or 0, 1),
            "o3": round(mv.o3 or 0, 1),
            "no2": round(mv.no2 or 0, 1),
        }
        max_ispu = max([mv.pm10, mv.pm25, mv.so2, mv.co, mv.o3, mv.no2])
        if max_ispu <= 50:
            today_data["status"] = "Baik"
        elif max_ispu <= 100:
            today_data["status"] = "Sedang"
        elif max_ispu <= 200:
            today_data["status"] = "Tidak Sehat"
        elif max_ispu <= 300:
            today_data["status"] = "Sangat Tidak Sehat"
        else:
            today_data["status"] = "Berbahaya"
    else:
        today_data = None
        
    if today_data:
        data7hari.insert(0, today_data)
    
    model = ModelMSSA.objects.filter(station=stasiun).order_by("-id_model").first()

    labels = [d["day"] for d in data7hari]
    chart_data = {
        "PM10": [d["pm10"] for d in data7hari],
        "PM25": [d["pm25"] for d in data7hari],
        "SO2": [d["so2"] for d in data7hari],
        "CO": [d["co"] for d in data7hari],
        "O3": [d["o3"] for d in data7hari],
        "NO2": [d["no2"] for d in data7hari],
    }

    if model:
        akurasi = f"{model.akurasi:.2f}%"
        mape = f"{(100 - model.akurasi):.2f}%"
        rmse = f"{model.rmse:.2f}"
        mse = f"{model.mse:.2f}"
        mae = f"{model.mae:.2f}"
        l = model.window_length
        k = model.embedding_dimension
        r = model.jumlah_komponen
        p = model.lag_data
    else:
        akurasi = "-"
        mape = "-"
        rmse = "-"
        mse = "-"
        mae = "-"
        l = "-"
        k = "-"
        r = "-"
        p = "-"

    last_update = data7hari[1]["day"] if data7hari else "-"
    
    # context = {
    #     "station": stasiun.nama_stasiun.replace(", Indonesia", ""),
    #     "data7hari": data7hari,
    #     "labels": json.dumps(labels),
    #     "chart_data": json.dumps(chart_data),
    #     "model": model,
    #     "akurasi": akurasi,
    #     "mape": mape,
    #     "rmse": rmse,
    #     "mse": mse,
    #     "mae": mae,
    #     "p": p,
    #     "l":l,
    #     "k":k,
    #     "r":r,
    #     "last_update": last_update,
    # }
    
    # return render(request, "prediksi.html", context)
    
    payload = {
        "station": stasiun.nama_stasiun.replace(", Indonesia", ""),
        "last_update": last_update,
        "data7hari": data7hari,
        "labels": labels,
        "chart_data": chart_data,
        "model": {
            "akurasi": akurasi,
            "mape": mape,
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "p": p,
            "l": l,
            "k": k,
            "r": r,
        }
    }
    
    if wants_json(request):
        return JsonResponse(payload, safe=False)
    
    payload["station_id"] = id_station
    
    payload["akurasi"] = payload["model"]["akurasi"]
    payload["mape"] = payload["model"]["mape"]
    payload["rmse"] = payload["model"]["rmse"]
    payload["mse"] = payload["model"]["mse"]
    payload["mae"] = payload["model"]["mae"]
    payload["p"] = payload["model"]["p"]
    payload["l"] = payload["model"]["l"]
    payload["k"] = payload["model"]["k"]
    payload["r"] = payload["model"]["r"]
    
    return render(request, "prediksi.html", payload)
