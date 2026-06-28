# 🤟 CNN-Spatial Attention-LSTM-Attention for BISINDO Recognition
### Real-Time Indonesian Sign Language Recognition on Android

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![Android](https://img.shields.io/badge/Android-API%2026+-brightgreen)
![ONNX](https://img.shields.io/badge/Export-ONNX-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📖 Overview

Penelitian ini mengembangkan sistem pengenalan gesture dinamis **Bahasa Isyarat Indonesia (BISINDO)** berbasis deep learning yang diimplementasikan secara real-time pada perangkat Android. Model yang diusulkan adalah **CNN-Spatial Attention-LSTM-Attention (CNNsa-LSTM)** berbasis ResNet18, yang dirancang untuk menyeimbangkan akurasi tinggi dan efisiensi komputasi agar dapat berjalan pada perangkat mobile dengan sumber daya terbatas.

> 📌 **Target**: Akurasi >90% dengan latency <100ms pada perangkat Android

---

## 🎯 Latar Belakang

Indonesia memiliki sekitar **1,92 juta individu Tuli**, namun hanya tersedia **~500 Juru Bahasa Isyarat (JBI) bersertifikat** hingga 2025. Ketimpangan ini menciptakan hambatan komunikasi signifikan di sektor pendidikan, kesehatan, dan layanan publik. Penelitian ini hadir sebagai solusi teknologi yang memungkinkan komunikasi mandiri dan real-time melalui smartphone.

---

## 🏗️ Arsitektur Model

```
Input Sequence (30 frames × 224×224×3)
         │
         ▼
┌─────────────────────┐
│   ResNet18 Backbone │  ← Ekstraksi fitur spasial per frame
│   (per-frame CNN)   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Spatial Attention │  ← Fokus pada region tangan yang relevan
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Bidirectional     │  ← Pemodelan temporal sequence gesture
│   LSTM              │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Temporal Attention │  ← Fokus pada frame paling informatif
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Classifier         │  ← Softmax → 40 kelas kata BISINDO
└─────────────────────┘
```

---

## 🔄 Pipeline Lengkap

```
📹 Raw Video Dataset (1600 videos)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 1. PRE-PROCESSING                                      │
│    Frame Extraction → Blur Filter → Resize (224×224)  │
│    Uniform Sampling (15fps) → Augmentation → Padding  │
│    Output: train/val/test.npz                         │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION                                  │
│    Mask R-CNN → Hand Segmentation per frame            │
│    MediaPipe  → 21 Keypoints × 3 (x,y,z) × 2 tangan  │
│    Output: sequences (N, 30, 126)                     │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING & EVALUATION                         │
│    CNN-Spatial Attention-LSTM-Attention (proposed)    │
│    vs Transformer (comparison)                        │
│    Metrics: Accuracy, F1, Precision, Recall,          │
│             Inference Time, Model Size, FLOPs         │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 4. EXPORT & DEPLOYMENT                                 │
│    PyTorch → ONNX → Android APK                       │
│    Real-time inference via CameraX + MediaPipe        │
└────────────────────────────────────────────────────────┘
```

---

## 📱 Android App Flow

```
┌──────────────────────────┐
│  ┌────────────────────┐  │
│  │                    │  │
│  │   [LIVE CAMERA]    │  │
│  │                    │  │
│  │  🖐️  hand tracked  │  │
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
│     [ 🔴 RECORD ]        │
└──────────────────────────┘
```

User menekan **RECORD** → app merekam 30 frame → inferensi → tampil **1 kata** hasil terjemahan.

---

## 📁 Struktur Direktori

```
CNN-LSTM_BISINDO_Android/
│
├── 📂 Datasets/
│   ├── WLBisindo/                   ← Raw video (1600 .mp4)
│   └── WLBisindo_Processed/         ← Output preprocessing (auto-generated)
│
├── 📂 PreProcessing/                ← Step 1: Video → Frame Sequences
│   ├── Config/
│   │   └── dataset_config.py        ← Konfigurasi dataset & preprocessing
│   ├── Scripts/
│   │   ├── frame_extractor.py       ← Ekstraksi frame dari video
│   │   ├── augmentor.py             ← Augmentasi & padding sequence
│   │   └── dataloader_builder.py    ← Split train/val/test → .npz
│   └── PreProcess.ipynb             ← Notebook controller
│
├── 📂 FeatureExtraction/            ← Step 2: Frame → Keypoint Sequences
│   ├── Config/
│   ├── Scripts/
│   │   ├── hand_segmentor.py        ← Mask R-CNN hand segmentation
│   │   ├── keypoint_extractor.py    ← MediaPipe 21 keypoints
│   │   └── sequence_builder.py      ← Build (N, 30, 126) sequences
│   └── FeatureExtraction.ipynb
│
├── 📂 Training/                     ← Step 3: Train & Compare Models
│   ├── Config/
│   ├── Models/
│   │   ├── cnn_spatial_attention_lstm.py   ← Proposed model
│   │   └── transformer.py                  ← Comparison model
│   ├── Scripts/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── comparator.py
│   └── Train.ipynb
│
├── 📂 Export/                       ← Step 4: PyTorch → ONNX
│   ├── export_onnx.py
│   ├── benchmark_onnx.py
│   └── Export.ipynb
│
├── 📂 AndroidApp/                   ← Step 5: Android Studio Project
│   └── [CameraX + MediaPipe + ONNX Runtime Mobile]
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
| Jumlah Kelas | 40 kata BISINDO |
| Jumlah Signer | 4 signer |
| Sample per Kelas | 10 per signer |
| Setelah Augmentasi | ~4.800 sequences |
| Split | 70% Train / 15% Val / 15% Test |

---

## ⚙️ Preprocessing Config

| Parameter | Value |
|---|---|
| Target FPS | 15 fps |
| Frame Size | 224 × 224 px |
| Sequence Length | 30 frames |
| Augmentasi | Flip, Brightness, Contrast, Rotation, Zoom |
| Augment Multiplier | 2× (1 original + 2 augmented) |
| Output Format | `.npz` (NumPy compressed) |

---

## 🆚 Model Comparison

| Metrik | CNNsa-LSTM (Proposed) | Transformer |
|---|---|---|
| Accuracy | - | - |
| F1-Score | - | - |
| Inference Time CPU | - | - |
| Model Size (MB) | - | - |
| FLOPs | - | - |
| Android Compatible | ✅ | ⚠️ |

> *Hasil akan diisi setelah eksperimen selesai*

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Deep Learning | PyTorch 2.3.0 |
| GPU | NVIDIA RTX 4060 (CUDA 12.1) |
| Computer Vision | OpenCV, MediaPipe |
| Hand Segmentation | Mask R-CNN (COCO pretrained) |
| Model Export | ONNX |
| Android Inference | ONNX Runtime Mobile |
| Android Camera | CameraX |
| Experiment Tracking | TensorBoard |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/username/CNN-LSTM_BISINDO_Android.git
cd CNN-LSTM_BISINDO_Android
pip install -r Requirements.txt
```

### 2. Siapkan Dataset
Letakkan video dataset di:
```
Datasets/WLBisindo/signer0_label0_sample1.mp4
                   signer0_label0_sample2.mp4
                   ...
```

### 3. Jalankan Preprocessing
```bash
cd PreProcessing
jupyter notebook PreProcess.ipynb
```

### 4. Feature Extraction
```bash
cd FeatureExtraction
jupyter notebook FeatureExtraction.ipynb
```

### 5. Training
```bash
cd Training
jupyter notebook Train.ipynb
```

---

## 📋 Requirements

```
torch==2.3.0+cu121
torchvision==0.18.0+cu121
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
onnx>=1.15.0
onnxruntime>=1.17.0
```

Install PyTorch dengan GPU support:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 📚 References

- Tang et al. (2022) — CNN-LSTM-Attention untuk prediksi adaptif
- Xu et al. (2024) — PBNet: Hybrid CNN-Transformer efisien
- Aljabar & Suharjito (2020) — BISINDO recognition framework
- Kindy et al. (2025) — Mobile deployment untuk sign language recognition

---

## 👤 Author

**[Nama Kamu]**
Thesis Project — [Nama Universitas]
[Tahun]

---

> ⚠️ **Status**: Project sedang dalam tahap pengembangan aktif. Semua kode telah ditulis namun belum dieksekusi/ditest.
