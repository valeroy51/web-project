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
                
                return JsonResponse({
                        "status": "ok",
                        "action": "save_pollutant",
                        "created": created,
                        "message": "Data polutan baru disimpan." if created else "Data polutan lama diperbarui.",
                    },status=200)


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
                
                return JsonResponse({
                        "status": "ok",
                        "action": "save_meteo",
                        "created": created,
                        "message": "Data meteorologi baru disimpan." if created else "Data meteorologi lama diperbarui.",
                    },status=200)

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
                    
                    return JsonResponse({"status": "error", "message": "Format file tidak didukung."}, status=400)
                
            except Exception as e:
                messages.error(request, f"Gagal membaca file: {e}")
                return JsonResponse({"status": "error", "message": "Gagal membaca file."}, status=400)
            
            if df.empty or df.dropna(how="all").shape[0] == 0:
                messages.error(request, "File kosong! Pastikan Anda sudah mengisi data sebelum mengunggah template.")
                return JsonResponse({"status": "error", "message": "File kosong!."}, status=400)

            if 'tanggal' not in [c.lower().strip().replace(' ', '') for c in df.columns]:
                messages.error(request, "File tidak memiliki kolom 'tanggal'. Pastikan format template sesuai.")
                return JsonResponse({"status": "error", "message": "File tidak memiliki kolom 'tanggal'."}, status=400)


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
                return JsonResponse({"status": "error", "message": "Kolom 'tanggal' tidak ditemukan di file."}, status=400)
            
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
                return JsonResponse({"status": "error", "message": "Tidak ada data dalam rentang tanggal valid."}, status=400)

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
                        return JsonResponse({"status": "error", "message": "Tidak ada kolom stasiun di file!."}, status=400)

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
                        return JsonResponse({"status": "error", "message": "Tidak ada kolom stasiun di file!."}, status=400)

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
            
            return JsonResponse({
                "status": "ok",
                "action": "upload",
                "file_name": uploaded_filename,
                "file_type": file_type,
                "inserted": inserted,
                "skipped": skipped,
                "upload_history": request.session.get("upload_history", []),
            },status=200)

            
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

    return JsonResponse({
            "status": "ok",
            "upload_history": context["upload_history"],
            "active_tab": context["active_tab"],
            "uploaded_filename": context["uploaded_filename"],
            "file_type": context["file_type"],
        },status=200)
