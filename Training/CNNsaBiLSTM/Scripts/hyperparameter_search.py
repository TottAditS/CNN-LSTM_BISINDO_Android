"""
Training/CNNsaBiLSTM/Scripts/hyperparameter_search.py
========================================================
Pencarian hyperparameter (grid kecil, dipilih manual berdasarkan alasan regularisasi)
untuk CNNsa-BiLSTM. Tujuan: cari kombinasi yang menggeneralisasi lebih baik (val
accuracy tinggi, GAP train-val kecil) dibanding konfigurasi produksi saat ini,
tanpa overfitting.

Tiap kali file ini dijalankan, itu SATU "run" baru -- disimpan ke folder tersendiri
bertimestamp (Training/Output/hyperparameter_search/runs/<YYYYMMDD_HHMMSS>/), TIDAK
menimpa run sebelumnya. Jadi folder runs/ itu sendiri jadi riwayat: berapa kali project
ini pernah menjalankan pencarian hyperparameter, kapan, dan hasilnya apa -- tinggal
lihat daftar foldernya.

Tiap run berisi (per trial, 11x per run):
    checkpoints/<nama_trial>.pt      bobot model di epoch val accuracy terbaik
    logs/<nama_trial>/                TensorBoard event log (SAMA formatnya dengan
                                       training utama trainer.py) -- lihat dengan:
                                       tensorboard --logdir Training/Output/hyperparameter_search/runs/<timestamp>/logs
    history/<nama_trial>.json         array lengkap per-epoch (train/val loss & acc, lr)
    search_results.json               ringkasan SEMUA trial di run itu (buat ranking cepat)

Cara pakai:
    Buka file ini di PyCharm (working directory = root project), klik Run.
    Semua trial jalan berurutan dalam SATU proses -- tidak perlu dipencet satu-satu.
    Checkpoint PRODUKSI (cnnsa_bilstm_best.pt) TIDAK disentuh.

Setelah selesai:
    1. Buka search_results.json di folder run yang baru -- berisi ranking tiap trial
       (val accuracy terbaik, epoch tercapainya, train accuracy akhir, gap train-val,
       path checkpoint).
    2. Pilih trial dengan val accuracy TERTINGGI *dan* gap train-val yang masih wajar
       (jangan pilih cuma karena val accuracy tertinggi kalau gap-nya jauh lebih besar
       dari baseline -- itu tanda mulai overfit, bukan perbaikan asli).
    3. Kalau ada trial yang lebih baik dari baseline: copy file checkpoint trial itu ke
       Training/Output/cnnsa_bilstm/checkpoints/cnnsa_bilstm_best.pt (TIMPA yang lama --
       backup dulu yang lama kalau mau jaga-jaga), lalu jalankan ulang
       Training/CNNsaBiLSTM/Scripts/evaluator.py untuk lihat angka test-set resminya
       (test set belum pernah dipakai sepanjang proses search ini -- baru dipakai SEKALI
       di akhir untuk konfigurasi yang benar-benar terpilih, supaya tidak bias).
    4. Kalau mau dipakai di aplikasi Android: jalankan ulang
       Export/Scripts/export_cnnsa_bilstm.py (otomatis baca model_config dari checkpoint
       yang baru), lalu copy manual cnnsa_bilstm.onnx yang baru ke
       BISINDO_ANDROID_APP/app/src/main/assets/.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Training.CNNsaBiLSTM.Config.train_config import TRAIN_NPZ, VAL_NPZ, MODEL_CONFIG as BASE_MODEL_CONFIG
from Training.CNNsaBiLSTM.Models.cnnsa_bilstm import CNNsaBiLSTM
from Training.CNNsaBiLSTM.Scripts.trainer import KeypointDataset
from Training.Scripts.device_utils import get_device

SEARCH_ROOT_DIR = Path(__file__).resolve().parents[3] / "Training" / "Output" / "hyperparameter_search"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = SEARCH_ROOT_DIR / "runs" / RUN_ID
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_DIR = RUN_DIR / "logs"
HISTORY_DIR = RUN_DIR / "history"
RESULTS_PATH = RUN_DIR / "search_results.json"

RANDOM_SEED = 42
MAX_EPOCHS = 50          # dibatasi dari 100 (default produksi) supaya total waktu search wajar
PATIENCE = 8             # sedikit lebih ketat dari 10 (default produksi), alasan sama
NUM_CLASSES = BASE_MODEL_CONFIG["num_classes"]
INPUT_SIZE = BASE_MODEL_CONFIG["input_size"]

# --- Grid trial: dipilih manual (bukan full cartesian product) supaya jumlah trial
# tetap wajar (~11x), tiap trial punya alasan regularisasi yang jelas. Trial 0 adalah
# baseline (konfigurasi produksi saat ini) sebagai titik pembanding.
TRIALS = [
    {"name": "baseline",              "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "dropout_0.4",           "dropout": 0.4, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "dropout_0.5",           "dropout": 0.5, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "wd_5e-4",               "dropout": 0.3, "weight_decay": 5e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "wd_1e-3",               "dropout": 0.3, "weight_decay": 1e-3, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "dropout_0.4_wd_5e-4",   "dropout": 0.4, "weight_decay": 5e-4, "hidden_size": 256, "learning_rate": 1e-3},
    {"name": "hidden_128",            "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 128, "learning_rate": 1e-3},
    {"name": "hidden_128_dropout_0.4","dropout": 0.4, "weight_decay": 1e-4, "hidden_size": 128, "learning_rate": 1e-3},
    {"name": "lr_5e-4",               "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 5e-4},
    {"name": "lr_2e-3",               "dropout": 0.3, "weight_decay": 1e-4, "hidden_size": 256, "learning_rate": 2e-3},
    {"name": "hidden_128_wd_5e-4",    "dropout": 0.4, "weight_decay": 5e-4, "hidden_size": 128, "learning_rate": 1e-3},
]


def run_trial(trial_cfg: dict, device: torch.device, train_ds, val_ds) -> dict:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0,
                               pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    model = CNNsaBiLSTM(
        input_size=INPUT_SIZE,
        feature_dim=BASE_MODEL_CONFIG["feature_dim"],
        hidden_size=trial_cfg["hidden_size"],
        num_layers=BASE_MODEL_CONFIG["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=trial_cfg["dropout"],
        attn_reduction=BASE_MODEL_CONFIG["attn_reduction"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_cfg["learning_rate"],
                                  weight_decay=trial_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    writer = SummaryWriter(log_dir=str(LOG_DIR / trial_cfg["name"]))

    best_val_acc = 0.0
    best_train_acc_at_best = 0.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = CHECKPOINT_DIR / f"{trial_cfg['name']}.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n{'='*70}\n[Trial] {trial_cfg['name']} -- {trial_cfg}\n{'='*70}")
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_loss = running_loss / total
        val_acc = correct / total
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"  Epoch {epoch:03d}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_train_acc_at_best = train_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "model_config": {
                    "input_size": INPUT_SIZE,
                    "feature_dim": BASE_MODEL_CONFIG["feature_dim"],
                    "hidden_size": trial_cfg["hidden_size"],
                    "num_layers": BASE_MODEL_CONFIG["num_layers"],
                    "num_classes": NUM_CLASSES,
                    "dropout": trial_cfg["dropout"],
                    "attn_reduction": BASE_MODEL_CONFIG["attn_reduction"],
                },
                "trial_hyperparams": trial_cfg,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping pada epoch {epoch} (patience {PATIENCE})")
            break

    writer.close()
    with open(HISTORY_DIR / f"{trial_cfg['name']}.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t0
    gap = best_train_acc_at_best - best_val_acc
    print(f"[Trial] {trial_cfg['name']} selesai dalam {elapsed:.1f}s -- "
          f"best val_acc={best_val_acc*100:.2f}% @ epoch {best_epoch}, "
          f"train_acc={best_train_acc_at_best*100:.2f}%, gap={gap*100:.2f} poin")

    return {
        "name": trial_cfg["name"],
        "hyperparams": trial_cfg,
        "best_val_acc": best_val_acc,
        "train_acc_at_best": best_train_acc_at_best,
        "gap_train_val": gap,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_acc"]),
        "elapsed_sec": elapsed,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(HISTORY_DIR / f"{trial_cfg['name']}.json"),
        "tensorboard_log_dir": str(LOG_DIR / trial_cfg["name"]),
    }


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[HyperparameterSearch] Run ID  : {RUN_ID}")
    print(f"[HyperparameterSearch] Run dir : {RUN_DIR}")
    print(f"[HyperparameterSearch] Device  : {device}")
    print(f"[HyperparameterSearch] {len(TRIALS)} trial akan dijalankan berurutan.")
    print(f"[HyperparameterSearch] Checkpoint produksi (cnnsa_bilstm_best.pt) TIDAK disentuh.")

    train_ds = KeypointDataset(TRAIN_NPZ)
    val_ds = KeypointDataset(VAL_NPZ)
    print(f"[HyperparameterSearch] Train: {len(train_ds)} sequence | Val: {len(val_ds)} sequence")

    results = []
    for trial_cfg in TRIALS:
        result = run_trial(trial_cfg, device, train_ds, val_ds)
        results.append(result)
        # Simpan progresif setelah tiap trial -- kalau proses terhenti di tengah jalan,
        # hasil trial yang sudah selesai tidak hilang.
        with open(RESULTS_PATH, "w") as f:
            json.dump({"run_id": RUN_ID, "trials": results}, f, indent=2)

    results_sorted = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)
    print(f"\n{'='*70}\n[HyperparameterSearch] RINGKASAN run {RUN_ID} (urut val accuracy tertinggi)\n{'='*70}")
    print(f"{'Trial':<22} {'Val Acc':>9} {'Train Acc':>10} {'Gap':>7} {'Epoch':>6}")
    for r in results_sorted:
        print(f"{r['name']:<22} {r['best_val_acc']*100:8.2f}% {r['train_acc_at_best']*100:9.2f}% "
              f"{r['gap_train_val']*100:6.2f}p {r['best_epoch']:6d}")

    print(f"\nHasil lengkap  : {RESULTS_PATH}")
    print(f"Checkpoint     : {CHECKPOINT_DIR}")
    print(f"History (JSON) : {HISTORY_DIR}")
    print(f"TensorBoard    : tensorboard --logdir \"{LOG_DIR}\"")
    print(f"\nSemua run tersimpan terpisah di {SEARCH_ROOT_DIR / 'runs'} -- riwayat berapa kali")
    print("search pernah dijalankan bisa dilihat dari daftar folder di situ.")
    print("\nIngat: checkpoint produksi (Training/Output/cnnsa_bilstm/checkpoints/cnnsa_bilstm_best.pt)")
    print("belum berubah. Baca instruksi promosi manual di docstring atas file ini.")


if __name__ == "__main__":
    main()
