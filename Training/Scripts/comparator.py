"""
Training/Scripts/comparator.py
==============================
Script pembanding untuk membandingkan model CNNsa-BiLSTM dengan model lain.
Saat ini berfungsi sebagai placeholder. Jika nanti model baru (misalnya Transformer)
telah dilatih, script ini akan memuat summary evaluasi masing-masing dan membuat tabel komparasi.
"""

import sys
import json
from pathlib import Path

# Setup path agar bisa import modul dari root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Training.Config.train_config import RESULTS_DIR


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


def compare_models():
    print("[Comparator] Memulai perbandingan model...")
    
    # 1. Muat hasil proposed model (CNNsa-BiLSTM)
    cnnsa_path = RESULTS_DIR / "evaluation_summary.json"
    cnnsa_summary = print_model_summary("CNNsa-BiLSTM (Proposed)", cnnsa_path)
    
    # 2. Muat hasil model pembanding (misalnya Transformer, jika ada)
    # Ganti path ini nanti setelah model pembanding dibuat
    transformer_path = RESULTS_DIR / "transformer_evaluation_summary.json"
    transformer_summary = print_model_summary("Transformer (Future)", transformer_path)
    
    if cnnsa_summary and transformer_summary:
        print("\n=== Tabel Perbandingan Akhir ===")
        print(f"{'Metrik':<25} | {'CNNsa-BiLSTM (Proposed)':<25} | {'Transformer (Future)':<25}")
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
    else:
        print("\n💡 Petunjuk:")
        print("  Belum ada model pembanding. Setelah Anda mengimplementasikan paper")
        print("  Transformer pilihan Anda, simpan hasil evaluasinya ke:")
        print(f"  {transformer_path}")
        print("  Lalu jalankan kembali script ini untuk membandingkan keduanya.")


if __name__ == "__main__":
    compare_models()
