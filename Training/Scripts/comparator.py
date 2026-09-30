"""
Training/Scripts/comparator.py
==============================
Script pembanding untuk membandingkan CNNsa-BiLSTM (proposed, Training/CNNsaBiLSTM/)
vs MViTv2 (baseline transformer, Training/MViTv2/). Model-agnostic: cukup baca dua
file *_evaluation_summary.json dari RESULTS_DIR (shared_config.py) yang ditulis
masing-masing evaluator, tidak import apapun spesifik ke satu model.

Selain print ke console, script ini juga MENYIMPAN hasilnya:
- Tabular : RESULTS_DIR / "model_comparison_summary.json"
- Visual  : CHARTS_DIR / "metrics_comparison.png" (bar chart Accuracy/Precision/Recall/F1)
            CHARTS_DIR / "training_curves_comparison.png" (kurva loss/accuracy kedua model,
            dibaca dari train_history.json & mvit_v2_train_history.json di RESULTS_DIR)
"""

import sys
import json
from pathlib import Path

# Setup path agar bisa import modul dari root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Training.Config.shared_config import RESULTS_DIR, CHARTS_DIR, create_results_dir


def print_model_summary(model_name: str, summary_path: Path):
    if not summary_path.exists():
        print(f"  [{model_name}] Summary tidak ditemukan di: {summary_path}")
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(f"\n=================== SUMMARY: {model_name} ===================")
    print(f"  Accuracy           : {summary.get('accuracy', 0)*100:.2f}%")
    print(f"  F1-Score (Weighted): {summary.get('f1_weighted', 0)*100:.2f}%")
    print(f"  Inference (CPU)    : {summary.get('avg_inference_cpu_ms', 0):.2f} ms/sequence")
    print(f"  Model Size         : {summary.get('model_size_mb', 0):.2f} MB")
    print(f"  Total Parameters   : {summary.get('total_params', 0):,}")
    print("=============================================================")
    return summary


def save_comparison_summary(cnnsa: dict, transformer: dict) -> Path:
    """Simpan tabel perbandingan (angka mentah + rasio) ke JSON, supaya tidak perlu
    disalin manual dari console lagi setiap kali menulis laporan."""
    comparison = {
        "cnnsa_bilstm": {
            "accuracy": cnnsa["accuracy"],
            "precision_weighted": cnnsa["precision_weighted"],
            "recall_weighted": cnnsa["recall_weighted"],
            "f1_weighted": cnnsa["f1_weighted"],
            "avg_inference_cpu_ms": cnnsa["avg_inference_cpu_ms"],
            "model_size_mb": cnnsa["model_size_mb"],
            "total_params": cnnsa["total_params"],
        },
        "mvit_v2": {
            "accuracy": transformer["accuracy"],
            "precision_weighted": transformer["precision_weighted"],
            "recall_weighted": transformer["recall_weighted"],
            "f1_weighted": transformer["f1_weighted"],
            "avg_inference_cpu_ms": transformer["avg_inference_cpu_ms"],
            "model_size_mb": transformer["model_size_mb"],
            "total_params": transformer["total_params"],
        },
        "ratios": {
            "accuracy_gap_points": (transformer["accuracy"] - cnnsa["accuracy"]) * 100,
            "param_ratio": transformer["total_params"] / cnnsa["total_params"],
            "size_ratio": transformer["model_size_mb"] / cnnsa["model_size_mb"],
            "latency_ratio": transformer["avg_inference_cpu_ms"] / cnnsa["avg_inference_cpu_ms"],
        },
    }

    out_path = RESULTS_DIR / "model_comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Comparator] Tabel perbandingan disimpan ke: {out_path}")
    return out_path


def plot_metrics_comparison(cnnsa: dict, transformer: dict) -> Path:
    """Bar chart Accuracy/Precision/Recall/F1 (weighted) CNNsa-BiLSTM vs MViTv2."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision\n(weighted)", "precision_weighted"),
        ("Recall\n(weighted)", "recall_weighted"),
        ("F1-Score\n(weighted)", "f1_weighted"),
    ]
    labels = [m[0] for m in metrics]
    cnnsa_vals = [cnnsa[m[1]] * 100 for m in metrics]
    mvit_vals = [transformer[m[1]] * 100 for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, cnnsa_vals, width, label="CNNsa-BiLSTM", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, mvit_vals, width, label="MViTv2", color="#ff7f0e")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Nilai (%)")
    ax.set_title(f"Perbandingan Performa Model -- CNNsa-BiLSTM vs MViTv2\n(data uji: 256 sequence, 32 kelas)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()

    out_path = CHARTS_DIR / "metrics_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Comparator] Chart perbandingan metrik disimpan ke: {out_path}")
    return out_path


def plot_training_curves() -> Path | None:
    """Kurva train/val loss & accuracy kedua model, dibaca dari train_history.json
    (CNNsa-BiLSTM) dan mvit_v2_train_history.json (MViTv2) di RESULTS_DIR."""
    import matplotlib.pyplot as plt

    cnnsa_path = RESULTS_DIR / "train_history.json"
    mvit_path = RESULTS_DIR / "mvit_v2_train_history.json"

    if not cnnsa_path.exists() or not mvit_path.exists():
        print(f"[Comparator] Training history tidak lengkap, skip plot_training_curves "
              f"(cari: {cnnsa_path.name}, {mvit_path.name})")
        return None

    with open(cnnsa_path) as f:
        cnnsa_hist = json.load(f)
    with open(mvit_path) as f:
        mvit_hist = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def _plot_pair(ax_loss, ax_acc, hist, title):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax_loss.plot(epochs, hist["train_loss"], label="Train Loss")
        ax_loss.plot(epochs, hist["val_loss"], label="Val Loss")
        ax_loss.set_title(f"{title} -- Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        train_acc_pct = [a * 100 for a in hist["train_acc"]]
        val_acc_pct = [a * 100 for a in hist["val_acc"]]
        ax_acc.plot(epochs, train_acc_pct, label="Train Acc")
        ax_acc.plot(epochs, val_acc_pct, label="Val Acc")
        best_idx = max(range(len(val_acc_pct)), key=lambda i: val_acc_pct[i])
        ax_acc.scatter([best_idx + 1], [val_acc_pct[best_idx]], color="red", zorder=5)
        ax_acc.annotate(f"Best: {val_acc_pct[best_idx]:.2f}%\n(epoch {best_idx + 1})",
                         xy=(best_idx + 1, val_acc_pct[best_idx]),
                         xytext=(10, -15), textcoords="offset points", fontsize=9)
        ax_acc.set_title(f"{title} -- Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

    _plot_pair(axes[0, 0], axes[0, 1], cnnsa_hist, "CNNsa-BiLSTM")
    _plot_pair(axes[1, 0], axes[1, 1], mvit_hist, "MViTv2")

    fig.suptitle("Kurva Training/Validation Loss dan Accuracy -- CNNsa-BiLSTM vs MViTv2", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = CHARTS_DIR / "training_curves_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Comparator] Chart kurva training disimpan ke: {out_path}")
    return out_path


def compare_models():
    print("[Comparator] Memulai perbandingan model...")
    create_results_dir()

    # 1. Muat hasil proposed model (CNNsa-BiLSTM)
    cnnsa_path = RESULTS_DIR / "evaluation_summary.json"
    cnnsa_summary = print_model_summary("CNNsa-BiLSTM (Proposed)", cnnsa_path)

    # 2. Muat hasil model pembanding (MViTv2, baseline transformer)
    transformer_path = RESULTS_DIR / "transformer_evaluation_summary.json"
    transformer_summary = print_model_summary("MViTv2 (Baseline)", transformer_path)

    if cnnsa_summary and transformer_summary:
        print("\n=== Tabel Perbandingan Akhir ===")
        print(f"{'Metrik':<25} | {'CNNsa-BiLSTM (Proposed)':<25} | {'MViTv2 (Baseline)':<25}")
        print("-" * 81)

        for key in ["accuracy", "f1_weighted", "avg_inference_cpu_ms", "model_size_mb"]:
            val1 = cnnsa_summary[key]
            val2 = transformer_summary[key]

            if "accuracy" in key or "f1" in key:
                print(f"{key:<25} | {val1*100:23.2f}% | {val2*100:23.2f}%")
            elif "ms" in key:
                print(f"{key:<25} | {val1:20.2f} ms | {val2:20.2f} ms")
            elif "mb" in key:
                print(f"{key:<25} | {val1:21.2f} MB | {val2:21.2f} MB")

        save_comparison_summary(cnnsa_summary, transformer_summary)
        plot_metrics_comparison(cnnsa_summary, transformer_summary)
        plot_training_curves()
    else:
        print("\n[Petunjuk]:")
        print("  Salah satu/kedua model belum dievaluasi. Jalankan:")
        print("  - Training/CNNsaBiLSTM/Train.ipynb  -> tulis evaluation_summary.json")
        print("  - Training/MViTv2/TrainMViT.ipynb   -> tulis transformer_evaluation_summary.json")
        print("  Lalu jalankan kembali script ini untuk membandingkan keduanya.")


if __name__ == "__main__":
    compare_models()
