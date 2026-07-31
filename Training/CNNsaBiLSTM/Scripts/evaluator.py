"""
Training/CNNsaBiLSTM/Scripts/evaluator.py
============================================
Script untuk mengevaluasi model yang sudah dilatih menggunakan test split.
- Mengalkulasi metrik: Accuracy, Precision, Recall, F1 (weighted & per-class)
- Membuat & menyimpan Confusion Matrix (seaborn heatmap)
- Menghitung waktu inferensi per sequence (CPU benchmark)
- Menghitung ukuran model (MB) dan jumlah parameter
- Menyimpan ringkasan evaluasi sebagai JSON (dibaca Training/Scripts/comparator.py)
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Setup path agar bisa import modul dari root (Scripts -> CNNsaBiLSTM -> Training -> ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Training.CNNsaBiLSTM.Config.train_config import (
    TEST_NPZ, MODEL_CONFIG, CHECKPOINT_DIR, CHECKPOINT_CONFIG, RESULTS_DIR,
    create_output_dirs
)
from Training.CNNsaBiLSTM.Scripts.trainer import KeypointDataset
from Training.CNNsaBiLSTM.Models.cnnsa_bilstm import CNNsaBiLSTM


class Evaluator:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device("cpu") # Evaluasi performa CPU untuk simulasi Android
        
        create_output_dirs()
        
        # Load test dataset
        print("\n[Evaluator] Loading test dataset...")
        self.test_dataset = KeypointDataset(TEST_NPZ)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        print(f"  Test samples: {len(self.test_dataset)}")
        
        # Load label map
        data = np.load(TEST_NPZ, allow_pickle=True)
        self.label_map = list(data["label_map"]) if "label_map" in data else [f"label_{i}" for i in range(MODEL_CONFIG["num_classes"])]
        
        # Initialize & load model
        print("\n[Evaluator] Loading trained model on CPU...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model = CNNsaBiLSTM(
            input_size=MODEL_CONFIG["input_size"],
            feature_dim=MODEL_CONFIG["feature_dim"],
            hidden_size=MODEL_CONFIG["hidden_size"],
            num_layers=MODEL_CONFIG["num_layers"],
            num_classes=MODEL_CONFIG["num_classes"],
            dropout=MODEL_CONFIG["dropout"],
            attn_reduction=MODEL_CONFIG["attn_reduction"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
    def get_model_size_mb(self) -> float:
        """Hitung ukuran file model checkpoint dalam MB."""
        return self.model_path.stat().st_size / (1024 * 1024)
        
    def count_parameters(self) -> tuple[int, int]:
        """Hitung total parameter dan parameter yang dilatih."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def benchmark_inference(self, num_runs: int = 100) -> float:
        """Benchmark latensi inferensi per sequence dalam milidetik di CPU."""
        dummy_input = torch.randn(1, 30, 126).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
            
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t_start = time.perf_counter()
                _ = self.model(dummy_input)
                t_end = time.perf_counter()
                times.append((t_end - t_start) * 1000) # ke milidetik
                
        avg_ms = np.mean(times)
        print(f"\n[Evaluator] Inference Time Benchmark (CPU):")
        print(f"  Rata-rata: {avg_ms:.2f} ms per sequence (target Android: < 100ms)")
        return float(avg_ms)

    def evaluate(self) -> dict:
        print("\n[Evaluator] Running evaluation over test set...")
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs, _ = self.model(X_batch)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 1. Hitung metrik keseluruhan
        acc = accuracy_score(all_targets, all_preds)
        f1_w = f1_score(all_targets, all_preds, average="weighted")
        prec_w = precision_score(all_targets, all_preds, average="weighted")
        rec_w = recall_score(all_targets, all_preds, average="weighted")
        
        print(f"\n[Evaluator] Hasil Evaluasi Akhir:")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  F1-Score  : {f1_w*100:.2f}%")
        print(f"  Precision : {prec_w*100:.2f}%")
        print(f"  Recall    : {rec_w*100:.2f}%")
        
        # 2. Detail klasifikasi per kelas
        report = classification_report(
            all_targets, all_preds, 
            target_names=self.label_map,
            output_dict=True,
            zero_division=0
        )
        
        # 3. Buat dan simpan Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        self._save_confusion_matrix_plot(cm)
        
        # 4. Model size & parameters
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
            "classification_report": report
        }
        
        # Simpan metrik ke JSON
        summary_path = RESULTS_DIR / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary disimpan ke: {summary_path}")
        
        return summary
        
    def _save_confusion_matrix_plot(self, cm: np.ndarray):
        """Simpan plot confusion matrix menggunakan matplotlib & seaborn."""
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
                yticklabels=self.label_map
            )
            plt.title("Confusion Matrix -- CNNsa-BiLSTM (Test Set)")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.tight_layout()
            
            plot_path = RESULTS_DIR / "confusion_matrix_cnnsa_bilstm.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Confusion Matrix tersimpan ke: {plot_path}")
        except ImportError:
            print("  [WARN] matplotlib / seaborn tidak terinstall, skip plotting confusion matrix.")


def run_evaluation():
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_CONFIG["filename"]
    if not checkpoint_path.exists():
        print(f"[Evaluator] ERROR: Model checkpoint tidak ditemukan di {checkpoint_path}")
        return
        
    evaluator = Evaluator(checkpoint_path)
    evaluator.evaluate()


if __name__ == "__main__":
    run_evaluation()
