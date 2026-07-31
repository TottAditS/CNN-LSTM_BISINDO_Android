# 🤟 CNNsa-BiLSTM for BISINDO Recognition
### Real-Time Indonesian Sign Language Recognition on Android

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![Android](https://img.shields.io/badge/Android-API%2026+-brightgreen)
![ONNX](https://img.shields.io/badge/Export-ONNX-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📖 Overview

Penelitian ini mengembangkan sistem pengenalan gesture dinamis **Bahasa Isyarat Indonesia (BISINDO)** berbasis deep learning yang diimplementasikan secara real-time pada perangkat Android. Model yang diusulkan adalah **CNN-Spatial Attention-BiLSTM-Attention (CNNsa-BiLSTM)** berbasis keypoint MediaPipe, yang dirancang untuk menyeimbangkan akurasi tinggi dan efisiensi komputasi agar dapat berjalan pada perangkat mobile dengan sumber daya terbatas.

> 📌 **Target**: Akurasi >90% dengan latency <100ms pada perangkat Android

---

## 🎯 Latar Belakang

Indonesia memiliki sekitar **1,92 juta individu Tuli**, namun hanya tersedia **~500 Juru Bahasa Isyarat (JBI) bersertifikat** hingga 2025. Ketimpangan ini menciptakan hambatan komunikasi signifikan di sektor pendidikan, kesehatan, dan layanan publik. Penelitian ini hadir sebagai solusi teknologi yang memungkinkan komunikasi mandiri dan real-time melalui smartphone.

---

## 🏗️ Arsitektur Model

Model CNNsa-BiLSTM menggunakan **keypoint MediaPipe** `(30, 126)` sebagai input — bukan raw image — sehingga jauh lebih ringan dan efisien untuk deployment Android.

```
Input Keypoints (30 frames × 126 features)
   126 = 2 tangan × 21 keypoint × 3 coords (x, y, z)
         │
         ▼
┌─────────────────────────┐
│   Per-frame CNN Block   │  ← Conv1D layers ekstraksi fitur dari 126 keypoint
│   (shared across frames)│    Output: (30, 256)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Spatial Attention     │  ← Fokus pada fitur keypoint yang paling relevan
│   (feature-wise)        │    Output: (30, 256) — weighted features
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Bidirectional LSTM    │  ← Pemodelan temporal maju & mundur
│   (2 layers, hidden=256)│    Output: (30, 512)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Temporal Attention    │  ← Fokus pada frame paling informatif dari 30 frame
│   (frame-wise)          │    Output: (512,) — context vector
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Classifier            │  ← FC(512→256) → ReLU → Dropout → FC(256→32) → Softmax
│                         │    Output: 32 kelas kata BISINDO
└─────────────────────────┘
```

---

## 🔄 Pipeline Lengkap

```
📹 Raw Video Dataset
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 1. PRE-PROCESSING                                        │
│    Scan Video → Blur Filter → Resize (224×224)           │
│    Uniform Sampling (15fps)                              │
│    Output: frame_manifest.json + frames.npy per video    │
│    Notebook: PreProcessing/PreProcess.ipynb               │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION                                    │
│    MediaPipe Hands → Palm Detection → 21 Keypoints       │
│    × 3 (x,y,z) × 2 tangan = 126 features/frame          │
│    Split Train/Val/Test → Augmentation (flip/rotate/zoom,│
│    train saja) → Temporal Pad/Crop → 30 frame fixed      │
│    Output: train/val/test_sequences.npz  (N, 30, 126)    │
│    Notebook: FeatureExtraction/FeatureExtraction.ipynb   │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 3a. TRAINING — CNNsa-BiLSTM (proposed model)             │
│    Input: keypoint sequences dari Step 2                 │
│    Metrics: Accuracy, F1, Precision, Recall,             │
│             Inference Time, Model Size                   │
│    Notebook: Training/CNNsaBiLSTM/Train.ipynb            │
└──────────────────────────────────────────────────────────┘
         │
         ▼ (paralel, bisa jalan tanpa Step 2 — baca frames.npy langsung dari Step 1)
┌──────────────────────────────────────────────────────────┐
│ 3b. TRAINING — MViTv2 (baseline transformer)             │
│    Input: raw video clip (16 frame, 224×224) dari Step 1 │
│    Fine-tune dari bobot pretrained Kinetics-400           │
│    Notebook: Training/MViTv2/TrainMViT.ipynb              │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 3c. PERBANDINGAN MODEL                                   │
│    Training/Scripts/comparator.py — video val/test SAMA  │
│    untuk kedua model (split di-share), tabel akurasi/F1/  │
│    latensi/ukuran model side-by-side                     │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 4. EXPORT KE ONNX ✅ (kedua model, PC saja)                │
│    PyTorch → ONNX, validasi output PyTorch vs ONNX RT     │
│    Notebook: Export/ExportModels.ipynb                    │
│    cnnsa_bilstm.onnx (14.6MB)  — kandidat Android          │
│    mvit_v2.onnx (132MB)        — pembanding riset saja     │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 5. DEPLOYMENT ANDROID (belum diimplementasi)               │
│    cnnsa_bilstm.onnx → Android APK                        │
│    Real-time inference via CameraX + MediaPipe on-device  │
│    (MViTv2 tidak dipakai — terlalu besar/lambat untuk      │
│     mobile, lihat bagian Model Evaluation)                 │
└──────────────────────────────────────────────────────────┘
```

---

## 📱 Android App Flow

```
┌──────────────────────────┐
│  ┌────────────────────┐  │
│  │                    │  │
│  │   [LIVE CAMERA]    │  │
│  │                    │  │
│  │    hand tracked    │  │
│  │  • landmark dots • │  │
│  └────────────────────┘  │
│                          │
│  Status: Ready...        │
│  ████████░░  24/30       │
│                          │
│  Result:  TERIMA KASIH   │
│                          │
│  History:                │
│  HALO | APA | KABAR      │
│                          │
│     [  RECORD  ]         │
└──────────────────────────┘
```

**Alur Inferensi Android:**
1. User tekan **RECORD**
2. CameraX merekam frame kamera
3. MediaPipe Hands ekstrak 126 keypoint per frame (secara langsung di Android)
4. Setelah 30 frame terkumpul → ONNX Runtime inferensi
5. Tampil **1 kata** hasil terjemahan

---

## 📁 Struktur Direktori

```
CNN-LSTM_BISINDO_Android/
│
├── 📂 Datasets/
│   ├── WLBisindo/                      ← Raw video (*.mp4)
│   ├── WLBisindo_Processed/            ← Output preprocessing (auto-generated)
│   │   ├── frames/                     ← frames.npy per video (bertingkat per label)
│   │   └── frame_manifest.json         ← metadata semua video
│   └── WLBisindo_Features/             ← Output feature extraction (auto-generated)
│       ├── keypoints/                  ← keypoint.npy per video asli (T, 126)
│       └── sequences/                  ← train/val/test_sequences.npz (N, 30, 126)
│                                          (train = original + augmented, val/test = original saja)
│
├── 📂 PreProcessing/                   ← Step 1: Video → Frame Sequences
│   ├── Config/
│   │   └── dataset_config.py           ← Konfigurasi dataset & preprocessing
│   ├── Scripts/
│   │   └── frame_extractor.py          ← Ekstraksi frame dari video
│   ├── PreProcess.ipynb                ← Notebook controller (jalankan ini)
│   └── PreProcessing Setup.md
│
├── 📂 FeatureExtraction/               ← Step 2: Frame → Keypoint Sequences (khusus CNNsa-BiLSTM)
│   ├── Config/
│   │   └── feature_config.py           ← Konfigurasi feature extraction & augmentasi
│   ├── Scripts/
│   │   ├── keypoint_extractor.py       ← MediaPipe → 21 keypoints per tangan
│   │   ├── keypoint_augmentor.py       ← Augmentasi keypoint (flip/rotate/zoom), train saja
│   │   ├── video_split.py              ← Split train/val/test SHARED (dipakai juga MViTv2)
│   │   └── sequence_builder.py         ← Split → augmentasi → (N, 30, 126) sequences
│   ├── FeatureExtraction.ipynb         ← Notebook controller (jalankan ini)
│   └── FeatureExtraction Setup.md
│
├── 📂 Training/                        ← Step 3: Train, Evaluasi & Bandingkan Model
│   ├── Config/
│   │   └── shared_config.py            ← RESULTS_DIR (SHARED, dibaca comparator.py)
│   ├── Scripts/
│   │   ├── comparator.py               ← Tabel perbandingan CNNsa-BiLSTM vs MViTv2
│   │   └── device_utils.py             ← get_device() (SHARED oleh kedua model)
│   ├── CNNsaBiLSTM/                    ← Proposed model (input: keypoint)
│   │   ├── Config/train_config.py      ← Hyperparameter & path
│   │   ├── Models/cnnsa_bilstm.py      ← Arsitektur CNNsa-BiLSTM
│   │   ├── Scripts/{trainer.py, evaluator.py}
│   │   └── Train.ipynb                 ← Notebook controller
│   ├── MViTv2/                         ← Baseline transformer (input: raw video, pretrained Kinetics-400)
│   │   ├── Config/mvit_config.py       ← Hyperparameter, freeze strategy, clip settings
│   │   ├── Models/mvit_v2.py           ← Wrapper torchvision mvit_v2_s
│   │   ├── Scripts/{mvit_dataset.py, mvit_trainer.py, mvit_evaluator.py}
│   │   └── TrainMViT.ipynb             ← Notebook controller
│   └── Training Setup.md
│
├── 📂 Export/                          ← Step 4: PyTorch → ONNX (kedua model, PC saja)
│   ├── Config/export_config.py         ← Path checkpoint/output, opset, shape input per model
│   ├── Scripts/
│   │   ├── export_cnnsa_bilstm.py      ← Export CNNsa-BiLSTM -> ONNX (legacy exporter)
│   │   ├── export_mvit_v2.py           ← Export MViTv2 -> ONNX (exporter baru, dynamo=True)
│   │   └── validate_onnx.py            ← Bandingkan output PyTorch vs ONNX Runtime
│   ├── ExportModels.ipynb              ← Notebook controller
│   ├── Export Setup.md
│   └── Output/onnx/                    ← cnnsa_bilstm.onnx, mvit_v2.onnx(.data) — auto-generated
│
└── Requirements.txt
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Nama | WLBisindo (Word-Level BISINDO) |
| Total Video | 1.600 video |
| Format | `signer{N}_label{N}_sample{N}.mp4` |
| Jumlah Kelas | 32 kata BISINDO (`label0`–`label31`) |
| Jumlah Signer | 5 signer (`signer0`–`signer4`) — `signer0` baru merekam 12/32 kelas |
| Sample per Kelas | Bervariasi (10 atau 15 per signer) |
| Split | 70% Train / 15% Val / 15% Test (per label, sebelum augmentasi) |
| Augmentasi | Hanya di train split, level keypoint (flip/rotate/zoom), 3x data train |

> ⚠️ Target awal riset adalah 40 kata / 4 signer (lihat tabel Training Config). Dataset yang
> sudah terkumpul saat ini baru 32 kelas dari 5 signer. `NUM_LABELS`/`NUM_CLASSES` di semua
> config sudah disesuaikan ke 32 — update kembali ke 40 jika dataset sudah lengkap.

---

## ⚙️ Preprocessing Config

| Parameter | Value |
|---|---|
| Target FPS | 15 fps |
| Frame Size | 224 × 224 px |
| Output Format | `.npy` per video + `frame_manifest.json` |

## 🔀 Feature Extraction — Split & Augmentation Config

| Parameter | Value |
|---|---|
| Sequence Length | 30 frames |
| Split | 70% Train / 15% Val / 15% Test (stratified per label, **sebelum** augmentasi) |
| Augmentasi | Flip, Rotation, Zoom — level keypoint, **hanya train split** |
| Augment Multiplier | 2 augmented + 1 original = 3x data train |
| Output Format | `.npz` (NumPy compressed) |

---

## 🏋️ Training Config

| Parameter | Value |
|---|---|
| Input Shape | (N, 30, 126) |
| Feature Dim (CNN out) | 256 |
| BiLSTM Hidden Size | 256 (× 2 = 512 bidirectional) |
| BiLSTM Layers | 2 |
| Dropout | 0.3 |
| Num Classes | 32 (sesuai dataset saat ini — lihat bagian Dataset) |
| Batch Size | 64 |
| Learning Rate | 1e-3 |
| Optimizer | Adam (weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau |
| Max Epochs | 100 |
| Early Stopping | patience=10 |

---

## 📈 Model Evaluation

Hasil aktual di test split dataset WLBisindo (1600 video asli, split signer-dependent 70/15/15,
video val/test identik untuk kedua model):

| Metrik | CNNsa-BiLSTM (Proposed) | MViTv2 (Baseline) |
|---|---|---|
| Accuracy | **94.53%** | 99.22% |
| F1-Score (weighted) | 94.51% | 99.22% |
| Precision (weighted) | 94.85% | 99.31% |
| Recall (weighted) | 94.53% | 99.22% |
| Inference Time (CPU) | **3.24 ms/sequence** | 400.07 ms/clip |
| Model Size | **43.94 MB** | 262.26 MB |
| Total Parameters | 3.84 juta | 34.25 juta |
| Android Compatible | ✅ | ❌ (ukuran & latensi jauh di luar budget mobile) |

> 📌 **Target awal tercapai**: CNNsa-BiLSTM mencapai akurasi >90% (94.53%) dengan latency
> <100ms (3.24ms) — sesuai target di bagian Overview.
>
> **Trade-off akurasi vs efisiensi**: MViTv2 unggul akurasi (+4.7 poin) karena model 9x lebih
> besar dan melihat raw video langsung (bukan keypoint), sejalan dengan hasil paper referensi
> (MViTv2 menang di setting signer-dependent). Tapi CNNsa-BiLSTM **~123x lebih cepat inferensi**
> dan **~6x lebih kecil** — inilah alasan utama CNNsa-BiLSTM yang dipilih untuk deployment
> Android, bukan MViTv2 yang jauh di luar budget model mobile.
>
> CNNsa-BiLSTM: training 34 epoch (early stopping dari max 100), ~48 detik di RTX 5060.
> MViTv2: training 30 epoch penuh (fine-tune dari Kinetics-400 pretrained), ~36 menit di RTX 5060.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Deep Learning | PyTorch 2.11.0 |
| GPU Training | NVIDIA GPU + CUDA 12.8 wheel (wajib untuk GPU Blackwell / RTX 50-series, backward-compatible untuk GPU lama) |
| Computer Vision | opencv-contrib-python, Pillow |
| Hand Keypoints | MediaPipe Hands (CPU-only di Windows — GPU delegate belum didukung Google di platform ini) |
| Model Export | ONNX |
| Android Inference | ONNX Runtime Mobile |
| Android Camera | CameraX |
| Experiment Tracking | TensorBoard |

> ⚠️ **GPU note**: training PyTorch (kedua model) otomatis pakai GPU kalau `torch.cuda.is_available()` = True
> (lihat `Training/Scripts/device_utils.py` -> `get_device()`, dipakai `CNNsaBiLSTM/Scripts/trainer.py` &
> `MViTv2/Scripts/mvit_trainer.py`, dipanggil di awal training/notebook dan akan print nama GPU + compute
> capability, atau warning jelas kalau jatuh ke CPU). Ekstraksi keypoint MediaPipe (`FeatureExtraction/`)
> selalu jalan di CPU terlepas dari ini — itu batasan resmi MediaPipe di Windows, bukan bug di repo ini.

---

## 🚀 Cara Menjalankan Pipeline (Step-by-Step)

> ✅ **Sudah divalidasi end-to-end**: seluruh urutan di bawah ini (Step 0-6) sudah dijalankan
> sungguhan — bukan cuma dibaca kodenya — memakai dataset sintetis kecil (video dummy), kode asli
> repo ini, dan environment asli (Python 3.11, PyTorch 2.11.0+cu128, mediapipe 0.10.21) di GPU
> NVIDIA RTX 50-series. Semua tahap (PreProcessing → FeatureExtraction → training kedua model →
> evaluasi kedua model → tabel perbandingan) berhasil selesai tanpa error di akhir validasi ini.

### Step 0 — Setup Environment

Buat environment khusus project ini (jangan pakai environment lain yang sudah terisi banyak
package — rawan bentrok versi):
```bash
conda create -n bisindo python=3.11 -y
conda activate bisindo
cd CNN-LSTM_BISINDO_Android
pip install -r Requirements.txt
```

> ⚠️ Kalau `conda activate` tidak mengubah prompt jadi `(bisindo) ...`, jalankan `conda init
> powershell` sekali lalu **tutup-buka ulang terminal** (bukan cuma tab baru).

Verifikasi environment benar-benar siap (jalankan ketiganya, semua harus sukses tanpa error):
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import mediapipe as mp; print(mp.solutions.hands.Hands())"
python -c "import cv2; print(cv2.__version__, cv2.VideoCapture)"
```
GPU harus `True` + nama GPU kamu muncul. Kalau salah satu di atas error, **jangan lanjut** —
lihat bagian "Requirements" di bawah, atau `Requirements.txt` langsung untuk catatan
troubleshooting lengkap (mediapipe/numpy/opencv sengaja dipin ke versi tertentu, ada
penjelasannya di komentar file itu).

> ⚠️ **Jangan install ulang package manapun** (`pip install ...`) di luar `Requirements.txt`
> setelah ini — termasuk jangan jalankan cell notebook manapun yang isinya `!pip install ...`.
> Menaikkan versi `numpy`/`opencv`/`mediapipe` secara terpisah akan membentur pin yang sudah
> disusun supaya kompatibel satu sama lain, dan bisa merusak `cv2`/`mediapipe` yang sudah jalan.

### Step 1 — Siapkan Dataset
Letakkan video di `Datasets/WLBisindo/`, format nama `signer{N}_label{N}_sample{N}.mp4`:
```
Datasets/WLBisindo/signer0_label0_sample1.mp4
                   signer0_label0_sample2.mp4
                   ...
```
Path ini otomatis terdeteksi oleh `PreProcessing/Config/dataset_config.py` — tidak perlu edit apapun.

### Step 2 — PreProcessing (extract frame mentah)
```bash
cd PreProcessing
jupyter notebook PreProcess.ipynb
```
Jalankan semua cell dari atas. Output: `Datasets/WLBisindo_Processed/frames/` + `frame_manifest.json`.
Untuk 1600 video asli, tahap ini butuh waktu tergantung jumlah & panjang video (bukan MediaPipe,
jadi relatif cepat — didominasi baca/resize video).

### Step 3 — Feature Extraction (khusus CNNsa-BiLSTM: keypoint MediaPipe, split, augmentasi)
```bash
cd FeatureExtraction
jupyter notebook FeatureExtraction.ipynb
```
Step 1 di notebook ini (ekstraksi keypoint MediaPipe) ~30-60 menit untuk 1600 video asli (CPU-only,
lihat catatan MediaPipe di README). Step 2 (split + augmentasi) ~1-5 menit. Output:
`Datasets/WLBisindo_Features/sequences/{train,val,test}_sequences.npz`.

### Step 4 — Training CNNsa-BiLSTM (proposed model) + evaluasi
```bash
cd Training/CNNsaBiLSTM
jupyter notebook Train.ipynb
```
Training loop (GPU otomatis kalau tersedia) → checkpoint terbaik → evaluasi test split → plot
loss/accuracy → confusion matrix. Output: `Training/Output/cnnsa_bilstm/checkpoints/`,
`Training/Output/results/evaluation_summary.json`.

### Step 5 — Training MViTv2 (baseline transformer) + evaluasi
Tidak perlu Step 3 — cukup Step 2 (baca `frames.npy` langsung):
```bash
cd Training/MViTv2
jupyter notebook TrainMViT.ipynb
```
Jauh lebih berat/lambat per-epoch dari Step 4 (model video 34.5M parameter, bukan keypoint).
Bobot pretrained Kinetics-400 (~140MB) di-download otomatis sekali di run pertama (tersimpan di
cache `~/.cache/torch/hub/`, tidak perlu download ulang di run berikutnya). Output:
`Training/Output/mvit_v2/checkpoints/`, `Training/Output/results/transformer_evaluation_summary.json`.

### Step 6 — Bandingkan Kedua Model
Cell terakhir `TrainMViT.ipynb` otomatis memanggil ini setelah Step 4 & 5 selesai. Bisa juga
dijalankan manual kapan saja setelah kedua model dievaluasi:
```bash
python Training/Scripts/comparator.py
```
Mencetak tabel accuracy / F1 / latensi CPU / ukuran model side-by-side, dibaca dari
`Training/Output/results/evaluation_summary.json` + `transformer_evaluation_summary.json`.

### Step 7 — Export ke ONNX
```bash
cd Export
jupyter notebook ExportModels.ipynb
```
Export kedua checkpoint ke ONNX + validasi otomatis (bandingkan output PyTorch vs ONNX Runtime
pakai sample test asli). Detail lengkap (kenapa dua model butuh exporter ONNX yang beda, kenapa
validasi CPU bukan GPU, dst) ada di `Export/Export Setup.md`. Output: `Export/Output/onnx/`.

### Belum Ada — Langkah Selanjutnya (di luar pipeline ini)
Aplikasi Android **belum diimplementasikan**. Setelah `cnnsa_bilstm.onnx` di-export & tervalidasi
(Step 7), langkah berikutnya adalah integrasi ke project Android baru (CameraX + ONNX Runtime
Mobile + MediaPipe Hands on-device) — belum ada kodenya di repo ini. `mvit_v2.onnx` **tidak**
diintegrasikan ke Android (terlalu besar/lambat untuk mobile), cuma untuk kelengkapan
dokumentasi/pembanding.

---

## 📋 Requirements

Daftar lengkap ada di `Requirements.txt` (`pip install -r Requirements.txt` install semuanya sekali jalan).
Inti dependency:

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
opencv-contrib-python==4.11.0.86  # JANGAN install opencv-python terpisah — konflik DLL di Windows
mediapipe==0.10.21                # dipin: versi >=0.10.30 (termasuk 1.0.0) menghapus API mp.solutions
Pillow>=10.0.0
numpy==1.26.4                     # dipin <2: mediapipe==0.10.21 mensyaratkan numpy<2
scikit-learn>=1.4.0
onnx>=1.17.0
onnxruntime>=1.20.0
```

> ⚠️ **mediapipe & numpy sengaja dipin ke versi lama** (bukan salah ketik): mulai mediapipe 0.10.30,
> Google menghapus total API `mediapipe.solutions` yang dipakai `keypoint_extractor.py` — hanya
> versi ≤0.10.21 yang masih punya `mp.solutions.hands`. Detail lengkap ada di komentar
> `Requirements.txt`.

Install PyTorch dengan GPU support (CUDA 12.8 — wajib untuk GPU Blackwell/RTX 50-series, tetap
backward-compatible untuk GPU NVIDIA generasi lama):
```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## 📚 References

- Tang et al. (2022) — CNN-LSTM-Attention untuk prediksi adaptif
- Aljabar & Suharjito (2020) — BISINDO recognition framework
- Kindy et al. (2025) — Mobile deployment untuk sign language recognition
- **Word-Level BISINDO: A Novel Video Indonesian Sign Language Dataset and Baseline Methods**
  (*Procedia Computer Science*, vol. 269, pp. 249–258, 2025) — sumber dataset WLBisindo yang
  dipakai repo ini, dan referensi baseline transformer (SPOTER, Siformer, **MViTv2** — MViTv2
  yang diimplementasikan di `Training/MViTv2/` sebagai pembanding CNNsa-BiLSTM).

---

## 👤 Author

**[Nama Kamu]**
Thesis Project — [Nama Universitas]
[Tahun]

---

> ✅ **Status**: Pipeline lengkap — PreProcessing → FeatureExtraction → Training (CNNsa-BiLSTM +
> baseline MViTv2) → Evaluasi → Perbandingan → **Export ke ONNX** — **sudah dijalankan
> end-to-end di dataset asli (1600 video)** dan hasilnya sudah ada (lihat bagian Model
> Evaluation & `Export/Export Setup.md`). Aplikasi Android **belum diimplementasikan** — itu
> langkah berikutnya.
