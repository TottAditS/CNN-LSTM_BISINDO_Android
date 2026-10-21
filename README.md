# CNNsa-BiLSTM for Word-Level BISINDO Recognition

Research code for a lightweight Indonesian Sign Language (BISINDO) recognizer that runs on
low-end Android phones. CNNsa-BiLSTM classifies sequences of MediaPipe hand keypoints. It is
compared with MViTv2-S, a video transformer that reads RGB frames, on the WL-BISINDO dataset.

The Android application used for the on-device tests is in a separate repository:
[BISINDO_ANDROID_APP](https://github.com/TottAditS/BISINDO_ANDROID_APP).

## Pipeline

```
Raw videos (WL-BISINDO)
  |
  1. PreProcessing       blur, resize to 224x224, uniform sampling at 15 fps
  |                      -> frames.npy per video + frame_manifest.json
  |
  +-------------------------------+
  |                               |
  2. FeatureExtraction            |  (MViTv2 reads the frames directly)
  |  MediaPipe Hands, 126 values  |
  |  per frame, normalization,    |
  |  split, augmentation,         |
  |  30-frame sequences           |
  |                               |
  3a. Train CNNsa-BiLSTM          3b. Fine-tune MViTv2-S
  |                               |
  +---------------+---------------+
                  |
  4. Compare both models on the same test videos
  5. Export both models to ONNX and validate against PyTorch
```

Both models use the same video-level split (70/15/15, stratified by class, seed 42), produced by
`FeatureExtraction/Scripts/video_split.py`.

## Model

Input: 30 frames x 126 values (2 hands x 21 landmarks x 3 coordinates). Each hand is expressed
relative to its wrist and divided by the wrist-to-middle-finger-MCP distance.

| Stage | Details | Output |
|---|---|---|
| Per-frame CNN | Conv1D 64 and 128 channels, kernel 5, BatchNorm, ReLU, MaxPool; linear projection; weights shared over frames | (30, 256) |
| Channel attention | Gate from the temporal average, reduction ratio 16 | (30, 256) |
| BiLSTM | 2 layers, 256 hidden units per direction | (30, 512) |
| Temporal attention | Softmax over time steps, weighted sum | (512) |
| Classifier | Dense 256, ReLU, dropout 0.3, 32 classes | (32) |

CNNsa-BiLSTM has 3.84 million parameters (15.4 MB as ONNX). MViTv2-S has 34.25 million
parameters (143 MB as ONNX); its last 4 of 16 blocks and a new 32-class head are fine-tuned
from Kinetics-400 weights.

## Results

Test split of WL-BISINDO, 256 videos (8 per class). Precision, recall, and F1 are weighted
averages. Latency is the mean per sample on a desktop CPU with PyTorch.

| Metric | CNNsa-BiLSTM | MViTv2-S |
|---|---|---|
| Accuracy | 95.31% | 99.22% |
| Precision | 95.56% | 99.31% |
| Recall | 95.31% | 99.22% |
| F1-score | 95.27% | 99.22% |
| Parameters | 3.84 M | 34.25 M |
| ONNX size | 15.4 MB | 143 MB |
| Desktop CPU latency | 7.42 ms | 400.07 ms |

On-device results (Xiaomi Poco M4 Pro, 160 attempts on replayed dataset clips; Samsung Galaxy
S24 Ultra, 31 attempts) are in `AndroidEvaluation/`; see [On-device data](#on-device-data).

## Repository layout

```
PreProcessing/
  Config/dataset_config.py          paths and frame settings
  Scripts/frame_extractor.py        video -> frames
  Scripts/visualize_preprocessing.py
  PreProcess.ipynb
FeatureExtraction/
  Config/feature_config.py          MediaPipe, sequence, split, and augmentation settings
  Scripts/keypoint_extractor.py     MediaPipe Hands -> (T, 126)
  Scripts/keypoint_augmentor.py     keypoint augmentation and Signer-Mixup
  Scripts/video_split.py            split shared by both models
  Scripts/sequence_builder.py       split -> augment -> (N, 30, 126) .npz
  Scripts/visualize_feature_extraction.py
  FeatureExtraction.ipynb
Training/
  Config/shared_config.py           shared results folder
  Scripts/comparator.py             comparison table and charts
  Scripts/device_utils.py           GPU selection
  CNNsaBiLSTM/                      config, model, trainer, evaluator, hyperparameter search, Train.ipynb
  MViTv2/                           config, model, dataset, trainer, evaluator, TrainMViT.ipynb
Export/
  Config/export_config.py
  Scripts/export_cnnsa_bilstm.py
  Scripts/export_mvit_v2.py
  Scripts/validate_onnx.py
  ExportModels.ipynb
AndroidEvaluation/                  per-attempt logs exported by the Android app
Requirements.txt
```

`Datasets/`, `Training/Output/`, and `Export/Output/` are created when the pipeline runs and are
not tracked.

## Reproducing the results

### 1. Environment

Python 3.11 on Windows 11 was used, with an NVIDIA RTX 5060 (8 GB) for training.

```bash
conda create -n bisindo python=3.11 -y
conda activate bisindo
pip install -r Requirements.txt
```

Check the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import mediapipe as mp; print(mp.solutions.hands.Hands())"
```

`mediapipe`, `numpy`, `opencv-contrib-python`, and `protobuf` are pinned to versions that work
together; installing newer versions separately breaks `mediapipe.solutions`. The reasons are
noted in `Requirements.txt`.

### 2. Dataset

Download WL-BISINDO from Kaggle:
<https://www.kaggle.com/datasets/glennleonali/wl-bisindo>

Place the videos anywhere under `Datasets/WLBisindo/`. File names must follow
`signer{S}_label{L}_sample{N}.mp4`, as in the original dataset. The dataset has 1,600 videos,
32 classes, and 5 signers.

### 3. Run the pipeline

Open each notebook from its own folder and run all cells in order.

| Step | Notebook | Output |
|---|---|---|
| 1 | `PreProcessing/PreProcess.ipynb` | `Datasets/WLBisindo_Processed/` |
| 2 | `FeatureExtraction/FeatureExtraction.ipynb` | `Datasets/WLBisindo_Features/sequences/*.npz` |
| 3 | `Training/CNNsaBiLSTM/Train.ipynb` | `Training/Output/cnnsa_bilstm/`, `evaluation_summary.json` |
| 4 | `Training/MViTv2/TrainMViT.ipynb` | `Training/Output/mvit_v2/`, `transformer_evaluation_summary.json`, comparison |
| 5 | `Export/ExportModels.ipynb` | `Export/Output/onnx/` |

Step 2 runs MediaPipe on the CPU and takes about 30-60 minutes for the 1,600 videos. Step 4
needs only step 1 and took about 36 minutes on an RTX 5060; the Kinetics-400 weights for
MViTv2 (about 140 MB) are downloaded automatically on the first run.

After step 1, the remaining steps can also be run as scripts from the repository root:

```bash
python FeatureExtraction/Scripts/keypoint_extractor.py
python FeatureExtraction/Scripts/sequence_builder.py
python Training/CNNsaBiLSTM/Scripts/trainer.py
python Training/CNNsaBiLSTM/Scripts/evaluator.py
python Training/MViTv2/Scripts/mvit_trainer.py
python Training/MViTv2/Scripts/mvit_evaluator.py
python Training/Scripts/comparator.py
python Export/Scripts/export_cnnsa_bilstm.py
python Export/Scripts/export_mvit_v2.py
python Export/Scripts/validate_onnx.py
```

Training curves can be viewed with
`tensorboard --logdir Training/Output/cnnsa_bilstm/logs` (or `mvit_v2/logs`).

Results may differ slightly from the table above because GPU training is not fully
deterministic.

### 4. Use the models in the Android app

Copy `cnnsa_bilstm.onnx`, `mvit_v2.onnx`, and `mvit_v2.onnx.data` from `Export/Output/onnx/`
to `app/src/main/assets/` of
[BISINDO_ANDROID_APP](https://github.com/TottAditS/BISINDO_ANDROID_APP) and build it as
described in that repository.

## Training settings

| | CNNsa-BiLSTM | MViTv2-S |
|---|---|---|
| Input | 30 x 126 keypoints | 16 x 224 x 224 RGB clip |
| Training data | 7,840 sequences (1,120 originals, 4,480 augmented, 2,240 Signer-Mixup) | 1,120 original videos, random horizontal flip |
| Optimizer | Adam, LR 1e-3, weight decay 1e-4 | Adam, LR 1e-5 backbone and 1e-3 head, weight decay 1e-4 |
| Batch size | 64 | 4, mixed precision |
| Max epochs | 100 | 30 |
| LR schedule | ReduceLROnPlateau on validation loss | ReduceLROnPlateau on validation loss |
| Early stopping | validation accuracy, patience 10 | validation accuracy, patience 7 |

Keypoint augmentation (training split only): horizontal flip (p = 0.5), rotation up to 20
degrees, zoom 0.8-1.2, Gaussian jitter (std 0.02), speed change 0.85-1.15, and hand dropout
(p = 0.15). Signer-Mixup interpolates two same-class sequences from different signers with a
weight drawn from 0.3-0.7.

## On-device data

The Android app exports one row per recording attempt.

| File | Device | Attempts |
|---|---|---|
| `AndroidEvaluation/poco_m4_pro_attempts.csv` | Xiaomi Poco M4 Pro (MediaTek Helio G96) | 160: 5 replayed dataset clips for each of the 32 classes |
| `AndroidEvaluation/s24_ultra_attempts.csv` | Samsung Galaxy S24 Ultra | 31: 17 replayed clips of Hari, Ingat, and Pagi, then 14 live attempts |

Columns: attempt number, target class, timestamp, raw and used frame counts, and for each model
the predicted label, confidence, inference latency (ms), process CPU usage (%), application
memory (PSS, KB), and whether the prediction was correct.

## Classes

| Index | Gloss | Index | Gloss | Index | Gloss | Index | Gloss |
|---|---|---|---|---|---|---|---|
| 0 | Air | 8 | Motor | 16 | Mengapa | 24 | Datang |
| 1 | Belajar | 9 | Saya | 17 | Bagaimana | 25 | Teman |
| 2 | Cari | 10 | Terima kasih | 18 | Merah | 26 | Keluarga |
| 3 | Hari | 11 | Tuli | 19 | Kuning | 27 | Rumah |
| 4 | Ingat | 12 | Apa | 20 | Hijau | 28 | Pagi |
| 5 | Lagi | 13 | Siapa | 21 | Hitam | 29 | Siang |
| 6 | Maaf | 14 | Kapan | 22 | Dengar | 30 | Sore |
| 7 | Makan | 15 | Di mana | 23 | Berangkat | 31 | Malam |

## Dataset reference

G. O. Kindy, G. Leonali, and H. Lucky, "Word-level BISINDO: A novel video Indonesian sign
language dataset and baseline methods," Procedia Computer Science, vol. 269, pp. 249-258, 2025,
doi: 10.1016/j.procs.2025.08.277.

## Authors

Totti Adithana Sunarto and Benfano Soewito, Master of Computer Science, BINUS Graduate Program,
Bina Nusantara University, Jakarta, Indonesia.
