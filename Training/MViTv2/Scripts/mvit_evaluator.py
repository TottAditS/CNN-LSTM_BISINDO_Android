"""
Training/MViTv2/Scripts/mvit_evaluator.py
============================================
Script untuk mengevaluasi MViTv2 yang sudah di-fine-tune, menggunakan test split.
- Mengalkulasi metrik: Accuracy, Precision, Recall, F1 (weighted & per-class)
- Membuat & menyimpan Confusion Matrix (seaborn heatmap)
- Menghitung waktu inferensi per clip (CPU benchmark)
- Menghitung ukuran model (MB) dan jumlah parameter
- Menyimpan ringkasan evaluasi sebagai "transformer_evaluation_summary.json" ke
  RESULTS_DIR (shared) -- path yang SAMA yang dibaca Training/Scripts/comparator.py,
  jadi comparator.py tidak perlu diubah sama sekali.

Catatan: benchmark CPU di sini murni untuk tabel perbandingan riset (skripsi),
BUKAN klaim MViTv2 layak deploy ke Android -- model ini >131MB dan jauh di atas
budget model mobile, beda dengan CNNsa-BiLSTM yang memang didesain untuk itu.
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Training.MViTv2.Config.mvit_config import (
    MODEL_CONFIG, CHECKPOINT_DIR, CHECKPOINT_CONFIG, RESULTS_DIR,
    CLIP_LENGTH, FRAME_SIZE,
    create_output_dirs,
)
from Training.MViTv2.Models.mvit_v2 import build_mvit_v2
from Training.MViTv2.Scripts.mvit_dataset import build_mvit_datasets


class MViTEvaluator:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device("cpu")  # Evaluasi performa CPU untuk tabel perbandingan

        create_output_dirs()

        print("\n[MViTEvaluator] Loading test dataset...")
        _, _, self.test_dataset = build_mvit_datasets()
        self.test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
        print(f"  Test samples: {len(self.test_dataset)}")

        self.label_map = [f"label_{i}" for i in range(MODEL_CONFIG["num_classes"])]

        print("\n[MViTEvaluator] Loading trained model on CPU...")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # pretrained=False -- bobot sesungguhnya datang dari checkpoint hasil fine-tune,
        # bukan dari Kinetics-400 lagi (hemat waktu, tidak perlu download ulang).
        self.model = build_mvit_v2(
            num_classes=MODEL_CONFIG["num_classes"],
            freeze_until_block=MODEL_CONFIG["freeze_until_block"],
            pretrained=False,
            dropout=MODEL_CONFIG["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def get_model_size_mb(self) -> float:
        return self.model_path.stat().st_size / (1024 * 1024)

    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def benchmark_inference(self, num_runs: int = 20) -> float:
        """Benchmark latensi inferensi per clip dalam milidetik di CPU (lebih sedikit run
        dari evaluator CNNsa-BiLSTM karena MViTv2 jauh lebih berat per forward pass)."""
        dummy_input = torch.randn(1, 3, CLIP_LENGTH, *FRAME_SIZE).to(self.device)

        for _ in range(3):
            _ = self.model(dummy_input)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t_start = time.perf_counter()
                _ = self.model(dummy_input)
                t_end = time.perf_counter()
                times.append((t_end - t_start) * 1000)

        avg_ms = np.mean(times)
        print(f"\n[MViTEvaluator] Inference Time Benchmark (CPU):")
        print(f"  Rata-rata: {avg_ms:.2f} ms per clip")
        return float(avg_ms)

    def evaluate(self) -> dict:
        print("\n[MViTEvaluator] Running evaluation over test set...")
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        acc = accuracy_score(all_targets, all_preds)
        f1_w = f1_score(all_targets, all_preds, average="weighted")
        prec_w = precision_score(all_targets, all_preds, average="weighted")
        rec_w = recall_score(all_targets, all_preds, average="weighted")

        print(f"\n[MViTEvaluator] Hasil Evaluasi Akhir:")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  F1-Score  : {f1_w*100:.2f}%")
        print(f"  Precision : {prec_w*100:.2f}%")
        print(f"  Recall    : {rec_w*100:.2f}%")

        report = classification_report(
            all_targets, all_preds,
            target_names=self.label_map,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(all_targets, all_preds)
        self._save_confusion_matrix_plot(cm)

        total_p, trainable_p = self.count_parameters()
        model_size = self.get_model_size_mb()
        avg_infer_ms = self.benchmark_inference()

        summary = {
            "accuracy": float(acc),
            "f1_weighted": float(f1_w),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "model_size_mb": float(model_size),
            "total_params": int(total_p),
            "trainable_params": int(trainable_p),
            "avg_inference_cpu_ms": float(avg_infer_ms),
            "classification_report": report,
        }

        # Path SAMA yang dibaca Training/Scripts/comparator.py -- zero perubahan di comparator.py
        summary_path = RESULTS_DIR / "transformer_evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary disimpan ke: {summary_path}")

        return summary

    def _save_confusion_matrix_plot(self, cm: np.ndarray):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(16, 14))
            sns.heatmap(
                cm,
                annot=False,
                fmt="d",
                cmap="Blues",
                xticklabels=self.label_map,
                yticklabels=self.label_map,
            )
            plt.title("Confusion Matrix -- MViTv2 (Test Set)")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.tight_layout()

            plot_path = RESULTS_DIR / "confusion_matrix_mvit_v2.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Confusion Matrix tersimpan ke: {plot_path}")
        except ImportError:
            print("  [WARN] matplotlib / seaborn tidak terinstall, skip plotting confusion matrix.")


def run_evaluation():
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_CONFIG["filename"]
    if not checkpoint_path.exists():
        print(f"[MViTEvaluator] ERROR: Model checkpoint tidak ditemukan di {checkpoint_path}")
        return

    evaluator = MViTEvaluator(checkpoint_path)
    evaluator.evaluate()


if __name__ == "__main__":
    run_evaluation()
