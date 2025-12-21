import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import joblib
from datetime import datetime
import time
from tqdm import tqdm
from django.utils import timezone

ENERGY_THR = 0.97
L_MIN, L_MAX = 40,100
N_JOBS_INNER = -1
LAMBDA_RIDGE = 1e-3

MERGE_DIR = os.getenv("MERGE_DIR")
NORM_DIR  = os.getenv("NORM_DIR")
OUT_ROOT  = os.getenv("OUT_ROOT")

SKIP_COLS = {"tanggal", "stasiun", "Tanggal", "Stasiun"}
POLUTAN   = {"pm10", "pm25", "so2", "no2", "co", "o3"}

FORCE_RETRAIN = True

os.makedirs(OUT_ROOT, exist_ok=True)

def _read_excel_any(path):
    return pd.read_excel(path)

def _standardize_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _pick_numeric_features(df):
    return [c for c in df.columns 
            if c not in SKIP_COLS 
            and np.issubdtype(df[c].dtype, np.number)]

def _load_norm_two_files(norm_pol_path, norm_met_path):
    df_pol = pd.read_excel(norm_pol_path)
    df_met = pd.read_excel(norm_met_path)
    df = pd.concat([df_pol, df_met], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]

    if not all(k in df.columns for k in ["kolom", "min", "max"]):
        raise ValueError("File normalisasi harus punya kolom: kolom, min, max")

    df["kolom"] = df["kolom"].astype(str).str.strip()
    return df

def denormalize_array(arr, fitur_cols, df_norm):
    arr = arr.copy()
    df_norm = df_norm.copy()
    df_norm.columns = [c.strip().lower() for c in df_norm.columns]

    for i, col in enumerate(fitur_cols):
        key = str(col).strip()
        if key not in df_norm["kolom"].values:
            print(f"Kolom '{col}' tidak ditemukan di normalisasi.")
            continue

        mn = float(df_norm.loc[df_norm["kolom"] == key, "min"].values[0])
        mx = float(df_norm.loc[df_norm["kolom"] == key, "max"].values[0])
        scale = mx - mn
        arr[:, i] = arr[:, i] * scale + mn

    return arr

def _format_now():
    return datetime.now().strftime("%d %B %Y, %H:%M")

def save_best_model(model_data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_data, save_path)
    print(f"Model terbaik disimpan di: {save_path}")

def load_best_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_model_to_db(station_name, bestL, bestK, bestr, bestp, total_mse, total_rmse, total_mae, akurasi):
    from API.models import Station, ModelMSSA

    try:
        station_obj = Station.objects.get(nama_stasiun__icontains=station_name)
    except Station.DoesNotExist:
        print(f"Station '{station_name}' tidak ditemukan di database.")
        return None

    deleted_count, _ = ModelMSSA.objects.filter(station=station_obj).delete()
    print(f"{deleted_count} model lama dihapus untuk {station_obj.nama_stasiun}")

    model_obj, created = ModelMSSA.objects.update_or_create(
        station = station_obj,
        window_length = bestL,
        embedding_dimension = bestK,
        jumlah_komponen = bestr,
        lag_data = bestp,
        mse = total_mse,
        rmse = total_rmse,
        mae = total_mae,
        akurasi = akurasi
    )

    print(f"Model MSSA tersimpan ke DB: ID={model_obj.id_model}")
    return model_obj

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

def save_prediction_to_db(model_obj, station_name, df_future):
    from API.models import PredictionResult, Station
    try:
        station_obj = Station.objects.get(nama_stasiun__icontains=station_name)
    except Station.DoesNotExist:
        print(f"Station '{station_name}' tidak ditemukan.")
        return

    base_date = timezone.now().date()

    for _, row in df_future.iterrows():

        pm10 = row.get("pm10")
        pm25 = row.get("pm25")
        so2  = row.get("so2")
        no2  = row.get("no2")
        co   = row.get("co")
        o3   = row.get("o3")

        nilai_max = max([pm10, pm25, so2, no2, co, o3])

        kategori = kategori_ispu(nilai_max)

        tanggal_pred = row["Tanggal"]

        PredictionResult.objects.update_or_create(
            station = station_obj,
            model = model_obj,
            tanggal_prediksi = tanggal_pred,

            pm10_pred = pm10,
            pm25_pred = pm25,
            so2_pred  = so2,
            no2_pred  = no2,
            co_pred   = co,
            o3_pred   = o3,

            indeks_kualitas_udara = kategori
        )

    print("Semua prediksi 7 hari berhasil disimpan ke DB.")

def fit_transition_matrix(X_hist, p, lam=None):

    if lam is None:
        lam = LAMBDA_RIDGE

    m, N = X_hist.shape
    if N < 2:
        raise ValueError("History terlalu pendek.")

    if p == 1:
        X_t    = X_hist[:, :-1]
        X_next = X_hist[:, 1:]

        G  = X_t @ X_t.T
        Hm = X_next @ X_t.T

        I  = np.eye(G.shape[0])
        A  = Hm @ np.linalg.pinv(G + lam * I)
        return A, m

    if N <= p:
        raise ValueError("History tak cukup untuk p>1.")

    X_t_list, X_next_list = [], []
    start = p - 1

    for t in range(start, N - 1):
        st  = np.concatenate([X_hist[:, t - lag]     for lag in range(p)])
        stp = np.concatenate([X_hist[:, t + 1 - lag] for lag in range(p)])
        X_t_list.append(st)
        X_next_list.append(stp)

    X_t    = np.array(X_t_list).T
    X_next = np.array(X_next_list).T

    G  = X_t @ X_t.T
    Hm = X_next @ X_t.T

    I  = np.eye(G.shape[0])
    A  = Hm @ np.linalg.pinv(G + lam * I)

    return A, m * p

def rolling_or_recursive_forecast(X_hist, A, p, actual_test=None, horizon=None):

    m = X_hist.shape[0]
    preds = []

    if p == 1:
        x_last = X_hist[:, -1:].copy()
    else:
        last_states = [X_hist[:, X_hist.shape[1] - 1 - lag] for lag in range(p)]
        x_last = np.concatenate(last_states).reshape(-1, 1)

    if actual_test is not None:
        H = len(actual_test)
    else:
        if horizon is None:
            raise ValueError("Kalau actual_test=None, wajib isi horizon.")
        H = horizon

    for t in range(H):
        x_state = A @ x_last
        x_pred  = x_state[:m].flatten()
        preds.append(x_pred)

        if actual_test is not None:
            y_next = actual_test[t]
        else:
            y_next = x_pred

        if p == 1:
            x_last = y_next.reshape(-1,1)
        else:
            x_last = np.vstack([
                y_next.reshape(-1,1),
                x_last[:m*(p-1)]
            ])

    return np.array(preds)


def multistep_forecast(X_hist, A, p, horizon):

    m, N = X_hist.shape
    preds = []

    if p == 1:
        x_last = X_hist[:, -1:].copy()
        for _ in range(horizon):
            x_pred = A @ x_last
            preds.append(x_pred.flatten())
            x_last = x_pred
        return np.array(preds)

    last_states = [X_hist[:, N - 1 - lag] for lag in range(p)]
    x_last = np.concatenate(last_states).reshape(-1, 1)

    for _ in range(horizon):
        x_state = A @ x_last
        x_pred  = x_state[:m]
        preds.append(x_pred.flatten())
        x_last = np.vstack([x_pred, x_last[:m*(p-1)]])

    return np.array(preds)


def run_mssa_for_L(L, df_train, df_test, fitur_cols, df_norm_all, energy_thr):
    hasil_local = []

    N_train = len(df_train)
    K = N_train - L + 1
    if K <= 0:
        return hasil_local

    print(f"\n==============================")
    print(f"Start L={L} | N={N_train} | K={K}")

    blok = []
    for col in fitur_cols:
        x = df_train[col].to_numpy()
        H = x[np.arange(L)[:, None] + np.arange(K)]
        blok.append(H)

    X = np.vstack(blok)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    energy = (S**2) / np.sum(S**2)
    r_star = int(np.argmax(np.cumsum(energy) >= energy_thr) + 1)
    r_star = max(1, r_star)

    print(f"r* (>={energy_thr}): {r_star}")

    M = len(blok)
    rec_cumul = [np.zeros(N_train) for _ in range(M)]

    for r in range(1, r_star + 1):
        if r >= L:
            break

        i = r - 1
        comp = S[i] * U[:, i][:, None] @ VT[i, :][None, :]

        for v in range(M):
            start, end = v * L, (v + 1) * L
            block = comp[start:end, :]
            Lb, Kb = block.shape
            Nb = Lb + Kb - 1
            res = np.zeros(Nb); cnt = np.zeros(Nb)
            for a in range(Lb):
                for b in range(Kb):
                    res[a+b] += block[a,b]; cnt[a+b] += 1
            rec_cumul[v] += (res / cnt)[:N_train]

        df_rec = pd.DataFrame(np.vstack(rec_cumul).T, columns=fitur_cols)
        X_hist = df_rec.to_numpy().T
        h = len(df_test)
        if X_hist.shape[1] < 2:
            continue

        for p in range(1, 8):
            try:
                A_p, dim = fit_transition_matrix(X_hist, p)
            except Exception as _:
                continue

            actual = df_test[fitur_cols].to_numpy()
            pred = rolling_or_recursive_forecast(
                X_hist,
                A_p,
                p,
                actual_test=actual
            )

            actual_dn = denormalize_array(actual, fitur_cols, df_norm_all)
            pred_dn   = denormalize_array(pred,   fitur_cols, df_norm_all)

            for idx, col in enumerate(fitur_cols):
                y_true = actual_dn[:, idx]
                y_pred = pred_dn[:, idx]

                mse = np.mean((y_pred - y_true)**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_pred - y_true))
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                akur = max(0, r2)*100

                hasil_local.append({
                    "L":L, "K":K, "r":r, "p":p, "Fitur":col,
                    "MSE":mse, "RMSE":rmse, "MAE":mae,
                    "R2":r2, "Akurasi(%)":akur
                })

    return hasil_local


def _plot_per_polutan(y_true, y_pred, title, save_path):
    plt.figure(figsize=(10,4.5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predict")
    plt.xlabel("Index Waktu (test set)")
    plt.ylabel("Konsentrasi (denormalized)")
    plt.title(title)
    plt.legend()
    _safe_mkdir(os.path.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_mssa_pipeline(DATA_PATH, NORM_POL_PATH, NORM_MET_PATH, OUT_DIR,
                      energy_thr, l_min, l_max, n_jobs_inner):

    time_total_start = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "Best_MSSA_Model.pkl")

    df = _read_excel_any(DATA_PATH)
    df = _standardize_cols(df)
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        df = df.sort_values("Tanggal").reset_index(drop=True)

    fitur_cols = _pick_numeric_features(df)
    if not fitur_cols:
        raise ValueError("Tidak ada kolom numerik.")

    N = len(df)
    idx_split = int(0.7 * N)
    df_train = df.iloc[:idx_split].reset_index(drop=True)
    df_test  = df.iloc[idx_split:].reset_index(drop=True)

    df_norm = _load_norm_two_files(NORM_POL_PATH, NORM_MET_PATH)

    print(f"\nDataset: {os.path.basename(DATA_PATH)}")
    print(f"Train={len(df_train)}, Test={len(df_test)}")

    saved = load_best_model(model_path)
    skip = False
    if saved and not FORCE_RETRAIN:
        print("Menggunakan model lama.")
        bestL = saved["L"]
        bestK = saved["K"]
        bestr = saved["r"]
        bestp = saved["p"]
        df_rec_final = saved["rekonstruksi_final"]
        skip = True

    tuning_time = None
    bestL_time = None
    if not skip:
        print("\nTuning MSSA...")
        time_tuning_start = time.time()
        
        max_possible_L = len(df_train) // 2
        l_max = min(l_max, max_possible_L)

        if l_min > l_max:
            print(f"l_min ({l_min}) lebih besar dari l_max ({l_max}). "
                f"l_min diset ulang menjadi {l_max}.")
            l_min = l_max

        if l_min == l_max:
            print(f"Hanya satu nilai L yang valid: {l_min}")
        
        res = Parallel(n_jobs=n_jobs_inner)(
            delayed(run_mssa_for_L)(L, df_train, df_test, fitur_cols, df_norm, energy_thr)
            for L in tqdm(range(l_min, l_max+1), desc="Loop L")
        )
        hasil = [x for sub in res for x in sub]
        if len(hasil) == 0:
            raise RuntimeError("Tidak ada hasil tuning (cek data & range L).")

        dfh = pd.DataFrame(hasil)

        dfh.to_excel(os.path.join(OUT_DIR, "Tuning_Results_All.xlsx"), index=False)

        dfh["fitur_lower"] = dfh["Fitur"].astype(str).str.lower()
        df_pol = dfh[dfh["fitur_lower"].isin(POLUTAN)].copy()

        if df_pol.empty:
            print("Tidak menemukan fitur polutan saat tuning, fallback pakai semua fitur.")
            df_pol = dfh.copy()

        agg = df_pol.groupby(["L","K","r","p"], as_index=False).agg(
            positif=("R2", lambda x: (x>0).sum()),
            med_r2=("R2","median"),
            mean_rmse=("RMSE","mean")
        )

        cand = agg.query("positif>=1").sort_values(
            ["positif","med_r2","mean_rmse"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        if cand.empty:
            cand = agg.sort_values(
                ["med_r2","mean_rmse"],
                ascending=[False, True]
            ).reset_index(drop=True)

        bestL = int(cand.loc[0,"L"])
        bestr = int(cand.loc[0,"r"])
        bestp = int(cand.loc[0,"p"])
        bestK = len(df_train) - bestL + 1

        time_tuning_end = time.time()
        tuning_time = time_tuning_end - time_tuning_start
        print(f"Lama TUNING MSSA: {tuning_time:.2f} detik")

        time_bestL_start = time.time()

        blok = []
        for col in fitur_cols:
            x = df_train[col].to_numpy()
            H = x[np.arange(bestL)[:,None] + np.arange(bestK)]
            blok.append(H)

        X = np.vstack(blok)
        U, S, VT = np.linalg.svd(X, full_matrices=False)

        M = len(blok)
        rec_final = [np.zeros(len(df_train)) for _ in range(M)]

        for i in range(bestr):
            comp = S[i]*U[:,i][:,None] @ VT[i,:][None,:]
            for v in range(M):
                start,end = v*bestL, (v+1)*bestL
                block = comp[start:end,:]
                Lb,Kb = block.shape
                Nb = Lb+Kb-1
                res = np.zeros(Nb); cnt=np.zeros(Nb)
                for a in range(Lb):
                    for b in range(Kb):
                        res[a+b]+=block[a,b]; cnt[a+b]+=1
                rec_final[v]+= (res/cnt)[:len(df_train)]

        df_rec_final = pd.DataFrame(np.vstack(rec_final).T, columns=fitur_cols)

        time_bestL_end = time.time()
        bestL_time = time_bestL_end - time_bestL_start
        print(f"Lama rekonstruksi untuk Best L={bestL}: {bestL_time:.2f} detik")

        model_data = {
            "L": bestL,
            "K": bestK,
            "r": bestr,
            "p": bestp,
            "fitur_cols": fitur_cols,
            "rekonstruksi_final": df_rec_final,
            "timestamp": _format_now()
        }
        save_best_model(model_data, model_path)
    else:
        tuning_time = None
        bestL_time = None

    time_eval_start = time.time()

    best = load_best_model(model_path)
    bestL = best["L"]; bestK = best["K"]
    bestr = best["r"]; bestp = best["p"]
    fitur_cols = best["fitur_cols"]
    df_rec_final = best["rekonstruksi_final"]

    X_hist = df_rec_final.to_numpy().T
    h = len(df_test)

    A_best, _ = fit_transition_matrix(X_hist, bestp)
    actual_raw = df_test[fitur_cols].to_numpy()
    pred_all = rolling_or_recursive_forecast(X_hist, A_best, bestp, actual_test=actual_raw)


    actual = df_test[fitur_cols].to_numpy()
    actual_dn = denormalize_array(actual, fitur_cols, df_norm)
    pred_dn   = denormalize_array(pred_all, fitur_cols, df_norm)

    eval_rows = []
    plot_dir = _safe_mkdir(os.path.join(OUT_DIR, "plots"))

    for i,col in enumerate(fitur_cols):
        col_l = str(col).lower()
        if col_l not in POLUTAN:
            continue

        y_true = actual_dn[:,i]
        y_pred = pred_dn[:,i]

        mse = np.mean((y_pred-y_true)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred-y_true))
        ssr = np.sum((y_true-y_pred)**2)
        sst = np.sum((y_true-np.mean(y_true))**2)
        r2  = 1 - (ssr/sst) if sst>0 else np.nan

        ttl = f"{col} — Actual vs Predict (R²={r2:.4f})"
        save_plot = os.path.join(plot_dir, f"{col}_Actual_vs_Predict.png")
        _plot_per_polutan(y_true, y_pred, ttl, save_plot)

        eval_rows.append([col, mse, rmse, mae, r2, save_plot])

    df_eval = pd.DataFrame(eval_rows, columns=["Fitur","MSE","RMSE","MAE","R2","PlotPath"])
    mean_acc = df_eval["R2"].clip(lower=0).mean()*100 if not df_eval.empty else 0.0
    df_eval.to_excel(os.path.join(OUT_DIR, "Eval_PerPolutan.xlsx"), index=False)

    time_eval_end = time.time()
    eval_time = time_eval_end - time_eval_start
    print(f"Lama evaluasi: {eval_time:.2f} detik")

    total_time = time.time() - time_total_start
    print(f"TOTAL waktu dataset ini: {total_time:.2f} detik")

    with open(os.path.join(OUT_DIR, "Timer_Log.txt"), "w") as f:
        f.write("=== LOG WAKTU (detik) ===\n")
        f.write(f"Tuning: {0.0 if tuning_time is None else tuning_time:.2f}\n")
        f.write(f"Rekonstruksi Best L: {0.0 if bestL_time is None else bestL_time:.2f}\n")
        f.write(f"Evaluasi: {eval_time:.2f}\n")
        f.write(f"TOTAL: {total_time:.2f}\n")

    if "Tanggal" in df.columns:
        last_date = df["Tanggal"].max().date()
    else:
        last_date = df["tanggal"].max().date()
    
    predict_next_7_days(
        best_model=best,
        df_norm=df_norm,
        fitur_cols=fitur_cols,
        save_dir=OUT_DIR,
        last_date=last_date
    )
    
    metrics = {}
    for pol in sorted(POLUTAN):
        row = df_eval[df_eval["Fitur"].str.lower() == pol]
        if not row.empty:
            metrics[f"R2_{pol.upper()}"]   = float(row["R2"].values[0])
            metrics[f"RMSE_{pol.upper()}"] = float(row["RMSE"].values[0])
            metrics[f"MAE_{pol.upper()}"]  = float(row["MAE"].values[0])
        else:
            metrics[f"R2_{pol.upper()}"]   = np.nan
            metrics[f"RMSE_{pol.upper()}"] = np.nan
            metrics[f"MAE_{pol.upper()}"]  = np.nan

    valid_rows = df_eval.copy()

    metrics["Total_MSE"]  = valid_rows["MSE"].mean() if not valid_rows.empty else np.nan
    metrics["Total_RMSE"] = valid_rows["RMSE"].mean() if not valid_rows.empty else np.nan
    metrics["Total_MAE"]  = valid_rows["MAE"].mean() if not valid_rows.empty else np.nan

    metrics["Total_R2"] = valid_rows["R2"].clip(lower=0).mean() if not valid_rows.empty else np.nan

    station_name = os.path.basename(DATA_PATH).split("_vs_")[0]
    model_db = save_model_to_db(
        station_name=station_name,
        bestL=bestL,
        bestK=bestK,
        bestr=bestr,
        bestp=bestp,
        total_mse=metrics["Total_MSE"],
        total_rmse=metrics["Total_RMSE"],
        total_mae=metrics["Total_MAE"],
        akurasi=mean_acc
    )

    if model_db is not None:
        df_future = pd.read_excel(os.path.join(OUT_DIR, "Prediksi_7Hari.xlsx"))
        save_prediction_to_db(model_db, station_name, df_future)

    return {
        "dataset": os.path.splitext(os.path.basename(DATA_PATH))[0],
        "L": bestL, "K": bestK,
        "r": bestr, "p": bestp,
        "Akurasi(%)": mean_acc,
        "Runtime_total": total_time,
        "Runtime_tuning": tuning_time,
        "Runtime_bestL": bestL_time,
        "Runtime_eval": eval_time,
        "Out": OUT_DIR,
        **metrics
    }

def predict_next_7_days(best_model, df_norm, fitur_cols, save_dir, last_date):

    print("\n=== PREDIKSI 7 HARI KE DEPAN ===")

    bestL = best_model["L"]
    bestK = best_model["K"]
    bestr = best_model["r"]
    bestp = best_model["p"]
    df_rec_final = best_model["rekonstruksi_final"]

    X_hist = df_rec_final.to_numpy().T

    A_best, _ = fit_transition_matrix(X_hist, bestp)

    h = 7
    pred_norm = rolling_or_recursive_forecast(
        X_hist,
        A_best,
        bestp,
        actual_test=None,
        horizon=h
    )

    pred_dn = denormalize_array(pred_norm, fitur_cols, df_norm)

    df_future = pd.DataFrame(pred_dn, columns=fitur_cols)

    tanggal_prediksi = [
        last_date + pd.Timedelta(days=i)
        for i in range(1, h + 1)
    ]
    df_future.insert(0, "Tanggal", tanggal_prediksi)

    save_path = os.path.join(save_dir, "Prediksi_7Hari.xlsx")
    df_future.to_excel(save_path, index=False)
    print(f"Prediksi 7 hari tersimpan di: {save_path}")

    plot_dir = os.path.join(save_dir, "plots_7d")
    os.makedirs(plot_dir, exist_ok=True)

    for col in fitur_cols:
        if str(col).lower() not in POLUTAN:
            continue

        plt.figure(figsize=(8,4))
        plt.plot(df_future["Tanggal"], df_future[col], marker="o")
        plt.title(f"Prediksi 7 Hari — {col}")
        plt.xlabel("Tanggal")
        plt.ylabel("Konsentrasi")
        plt.grid(True)

        plt.savefig(
            os.path.join(plot_dir, f"{col}_7hari.png"),
            dpi=150
        )
        plt.close()

    print(f"Grafik prediksi 7 hari disimpan di: {plot_dir}")

    return df_future


def main():
    files = glob.glob(os.path.join(MERGE_DIR,"*.xlsx"))
    if not files:
        print("Tidak ada file.")
        return

    master = []

    for path in files:
        nama = os.path.splitext(os.path.basename(path))[0]
        print("\n===============================")
        print(f"Dataset: {nama}")
        print("===============================")

        if "_vs_" not in nama:
            print("Format salah (harus _vs_) > skip")
            continue

        s1, s2 = nama.split("_vs_",1)
        norm_pol = os.path.join(NORM_DIR, f"{s1}_MinMax.xlsx")
        norm_met = os.path.join(NORM_DIR, f"{s2}_MinMax.xlsx")

        out = os.path.join(OUT_ROOT, nama)
        os.makedirs(out, exist_ok=True)

        try:
            rec = run_mssa_pipeline(
                DATA_PATH=path,
                NORM_POL_PATH=norm_pol,
                NORM_MET_PATH=norm_met,
                OUT_DIR=out,
                energy_thr=ENERGY_THR,
                l_min=L_MIN, l_max=L_MAX,
                n_jobs_inner=N_JOBS_INNER
            )
            master.append(rec)
        except Exception as e:
            print(f"Error di {nama}: {e}")

    if master:
        dfm = pd.DataFrame(master)

        base_cols = ["dataset","Out","L","K","r","p","Akurasi(%)",
                     "Runtime_total","Runtime_tuning","Runtime_bestL","Runtime_eval"]
        pol_cols  = []
        for pol in sorted(POLUTAN):
            pol_cols += [f"R2_{pol.upper()}", f"RMSE_{pol.upper()}", f"MAE_{pol.upper()}"]

        ordered = [c for c in base_cols if c in dfm.columns] + [c for c in pol_cols if c in dfm.columns]
        ordered += [c for c in dfm.columns if c not in ordered]

        dfm = dfm[ordered]
        out_master = os.path.join(OUT_ROOT,"Master_Summary.xlsx")
        dfm.to_excel(out_master, index=False)
        print("\nMASTER SUMMARY tersimpan di:")
        print(out_master)
        print(dfm)
    else:
        print("\nTidak ada hasil.")


if __name__ == "__main__":
    main()
