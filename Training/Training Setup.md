# 🏋️ Training Setup Guide (CNNsa-BiLSTM)

Panduan langkah demi langkah untuk melakukan training model **CNNsa-BiLSTM** pada dataset BISINDO yang telah diekstrak fiturnya.

---

## 📁 Struktur File Training

Direktori `Training/` terstruktur sebagai berikut:
- **`Config/train_config.py`**: Berisi konfigurasi path dan hyperparameter terpusat.
- **`Models/cnnsa_bilstm.py`**: Implementasi PyTorch dari arsitektur model usulan.
- **`Scripts/trainer.py`**: Script loop training utama (dengan checkpoints, early stopping, dan learning rate scheduling).
- **`Scripts/evaluator.py`**: Mengalkulasi metrik performa model pada test set, memplot confusion matrix, dan melakukan benchmark latensi CPU.
- **`Scripts/comparator.py`**: Script pembanding baseline (placeholder untuk model baru seperti Transformer).
- **`Train.ipynb`**: Notebook controller utama untuk menjalankan seluruh proses pelatihan, evaluasi, dan plotting hasil secara interaktif.

---

## 🚀 Langkah 1: Persiapan Requirements

Pastikan seluruh package pendukung training terinstall. Kami menyarankan menggunakan lingkungan conda atau virtualenv dengan Python >= 3.9.

Jika Anda menggunakan CUDA GPU (sangat disarankan untuk mempercepat training):
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Instalasi dependency visualisasi dan evaluasi:
```bash
pip install matplotlib seaborn scikit-learn tensorboard
```

---

## 🏃 Langkah 2: Cara Menjalankan Training

Anda memiliki dua opsi untuk melatih model:

### Opsi A: Menggunakan Jupyter Notebook (Sangat Disarankan)
1. Buka terminal di root project dan aktifkan Jupyter Notebook:
   ```bash
   cd Training
   jupyter notebook Train.ipynb
   ```
2. Jalankan cell demi cell secara berurutan:
   - **Cell 1 & 2**: Cek integritas file input `.npz` dan setup device (CPU/GPU).
   - **Cell 3 & 4**: Inisialisasi arsitektur model dan tampilkan model summary.
   - **Cell 5 & 6**: Jalankan training loop. Proses log epoch akan dicetak langsung di bawah cell.
   - **Cell 7 & 8**: Evaluasi model terbaik pada test split.
   - **Cell 9 & 10**: Plot kurva loss & accuracy, serta tampilkan grafik confusion matrix secara visual.

### Opsi B: Menggunakan Terminal Command Line
Jika Anda ingin melatih model di latar belakang (background task):
1. **Latih Model**:
   ```bash
   python Scripts/trainer.py
   ```
2. **Evaluasi Hasil**:
   ```bash
   python Scripts/evaluator.py
   ```

---

## 📊 Hasil Output Training (Auto-Generated)

Setelah training dan evaluasi selesai dijalankan, direktori **`Training/Output/`** akan dibuat secara otomatis dengan isi sebagai berikut:

```
Training/Output/
├── 📂 checkpoints/
│   └── cnnsa_bilstm_best.pt        ← Bobot model terbaik berdasarkan Val Accuracy
├── 📂 logs/
│   └── events.out.tfevents.*       ← Log TensorBoard untuk memantau loss & metrics
└── 📂 results/
    ├── train_history.json          ← Log loss & accuracy per epoch
    ├── evaluation_summary.json     ← Metrik presisi, recall, F1, & latensi CPU
    └── confusion_matrix.png        ← Plot visualisasi confusion matrix (heatmap)
```

---

## 📈 Monitoring dengan TensorBoard

Untuk melihat visualisasi proses training secara interaktif dan real-time:
```bash
tensorboard --logdir=Output/logs
```
Buka browser Anda dan navigasikan ke `http://localhost:6006/` untuk melihat kurva loss, akurasi, dan pergerakan learning rate.

---

## ⚙️ Cara Menyesuaikan Hyperparameter

Semua parameter latihan terpusat di **`Config/train_config.py`**. Anda bisa mengedit nilai di dalamnya langsung:
- **`batch_size`**: Default `64`. Kurangi menjadi `32` jika Anda mendapatkan error *Out of Memory (OOM)* pada GPU.
- **`learning_rate`**: Default `1e-3`. Gunakan nilai yang lebih kecil jika kurva loss berfluktuasi terlalu tajam.
- **`patience`** (Early Stopping): Default `10` epoch. Menentukan seberapa lama sistem menunggu performa validasi membaik sebelum menghentikan training lebih cepat untuk mencegah overfitting.
- **`hidden_size`**: Default `256`. Mengontrol dimensi internal layer BiLSTM.
