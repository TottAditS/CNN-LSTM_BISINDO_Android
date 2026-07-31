# 📦 Export Setup Guide (PyTorch → ONNX)

Konversi kedua model (CNNsa-BiLSTM & MViTv2) dari checkpoint PyTorch (`.pt`) ke format **ONNX**
(`.onnx`) — format model universal yang bisa dijalankan tanpa PyTorch, termasuk nanti di Android
lewat ONNX Runtime Mobile.

Ini **bukan** app Android — cuma tahap konversi + validasi di PC. Project Android (CameraX +
MediaPipe on-device + ONNX Runtime Mobile) belum diimplementasikan, itu langkah berikutnya.

> ✅ **Sudah divalidasi end-to-end** memakai checkpoint hasil training asli (bukan dummy) dan
> sample nyata dari test split — lihat hasil di bagian "Hasil Validasi" di bawah.

---

## 📁 Struktur

```
Export/
├── Config/
│   └── export_config.py        ← Path checkpoint/output, opset ONNX, shape input
├── Scripts/
│   ├── export_cnnsa_bilstm.py  ← Export CNNsa-BiLSTM -> ONNX
│   ├── export_mvit_v2.py       ← Export MViTv2 -> ONNX
│   └── validate_onnx.py        ← Bandingkan output PyTorch vs ONNX Runtime
├── ExportModels.ipynb          ← Notebook controller (jalankan ini)
├── Export Setup.md
└── Output/                     ← Auto-generated
    └── onnx/
        ├── cnnsa_bilstm.onnx
        ├── mvit_v2.onnx        ← cuma "peta graph", bobotnya di file sebelah (lihat catatan)
        └── mvit_v2.onnx.data   ← bobot MViTv2 (~140MB, WAJIB ada di folder yang sama)
```

## Prasyarat

Checkpoint kedua model sudah ada (hasil `Training/CNNsaBiLSTM/Train.ipynb` dan
`Training/MViTv2/TrainMViT.ipynb`):
- `Training/Output/cnnsa_bilstm/checkpoints/cnnsa_bilstm_best.pt`
- `Training/Output/mvit_v2/checkpoints/mvit_v2_best.pt`

Dependency: `torch` (include `torch.onnx`), `onnxruntime`, dan `onnxscript` (khusus export
MViTv2 — lihat "Detail Teknis") — semua sudah ada di `Requirements.txt` root repo.

## Cara Menjalankan

```bash
cd Export
jupyter notebook ExportModels.ipynb
```
Jalankan cell dari atas: cek checkpoint → export CNNsa-BiLSTM → export MViTv2 → validasi
(PyTorch vs ONNX Runtime, kedua model).

Atau lewat command line:
```bash
python Export/Scripts/export_cnnsa_bilstm.py
python Export/Scripts/export_mvit_v2.py
python Export/Scripts/validate_onnx.py
```

## Detail Teknis

**Dua model butuh exporter ONNX yang BEDA** — dikonfirmasi lewat tes langsung, bukan tebakan:

| | CNNsa-BiLSTM | MViTv2 |
|---|---|---|
| Exporter | Legacy (TorchScript-based, `dynamo=False`) | Baru (`torch.export`-based, `dynamo=True`) |
| Opset | 17 | 18 |
| Kenapa | Sudah cukup & tervalidasi | Exporter legacy **"sukses" export tapi GAGAL dijalankan** di ONNX Runtime (`Loop subgraph input 0 has unknown shape`) — pooling-attention MViT ke-trace jadi Loop node dinamis yang shape-nya tidak bisa di-infer exporter lama |
| Dependency tambahan | - | `onnxscript` (exporter baru butuh ini) |
| Format output | 1 file `.onnx` (bobot di dalam) | 2 file: `.onnx` (graph) + `.onnx.data` (bobot, ~140MB, **harus** satu folder) |

Detail lain:
- **Batch size difix ke 1** (bukan dynamic) — sesuai use-case nyata (inferensi satu
  sequence/clip sekali jalan di HP, bukan batch). Juga menghindari warning PyTorch soal
  kombinasi LSTM + dynamic batch size pada CNNsa-BiLSTM.
- **model_config diambil dari checkpoint itu sendiri** (bukan re-import dari
  `Training/*/Config/`) — supaya arsitektur yang dibangun ulang selalu cocok persis dengan yang
  benar-benar dilatih, walau config berubah setelahnya.
- Exporter baru (`dynamo=True`) mencetak progress pakai karakter unicode (✅) yang bisa
  `UnicodeEncodeError` di konsol Windows non-UTF8 — `export_mvit_v2.py` sudah
  `sys.stdout.reconfigure(encoding="utf-8")` di awal supaya aman dijalankan dari console manapun.

## ⚠️ Kenapa Validasi Pakai CPU, Bukan GPU

`onnxruntime` (CPU-only) sudah cukup dan sudah ada di `Requirements.txt`. Untuk GPU
(`onnxruntime-gpu`) sempat dicoba tapi **tidak dilanjutkan** karena versi terbaru
(1.28.0) butuh **CUDA 13 + cuDNN 9** system libraries yang belum terpasang di mesin ini (beda
dari PyTorch yang bundle CUDA 12.8 sendiri di dalam wheel) — `CUDAExecutionProvider`-nya diam-diam
jatuh ke CPU kalau library itu tidak ada (`cublasLt64_13.dll` missing, dikonfirmasi lewat tes
langsung). Daripada mengejar dependency CUDA 13 terpisah untuk manfaat yang kecil, validasi
sengaja pakai CPU (`CPUExecutionProvider`) — ini juga lebih relevan ke kondisi target
deployment (Android umumnya jalan CPU/NNAPI, bukan CUDA ala desktop).

Kalau suatu saat mau coba lagi pakai GPU: `pip install "onnxruntime-gpu[cuda,cudnn]==1.28.0"`
lalu cek `onnxruntime.InferenceSession(..., providers=["CUDAExecutionProvider", ...]).get_providers()`
— kalau hasil pertamanya bukan `CUDAExecutionProvider`, berarti diam-diam fallback ke CPU lagi.

## ✅ Hasil Validasi (data test asli, CPU)

`validate_onnx.py` mencetak `PASS`/`FAIL` per model berdasarkan selisih output PyTorch vs ONNX
(threshold `ATOL = 1e-4`) **dan** kecocokan kelas prediksi (argmax) — dua-duanya harus cocok.
Hasil aktual (checkpoint asli, sample nyata dari test split):

| Metrik | CNNsa-BiLSTM | MViTv2 |
|---|---|---|
| Status | **PASS** | **PASS** |
| Selisih maks vs PyTorch | 0.0000038 | 0.0000134 |
| Prediksi kelas sama? | Ya | Ya |
| Ukuran ONNX | 14.65 MB | 132.20 MB (2.56 + 140.31 file `.data`) |
| Latensi PyTorch (CPU) | 10.5 ms | 422.5 ms |
| Latensi ONNX Runtime (CPU) | **2.9 ms** (lebih cepat dari PyTorch) | 2408.0 ms (lebih **lambat** dari PyTorch) |

> MViTv2 di ONNX Runtime lebih lambat dari PyTorch native — wajar untuk model attention
> kompleks yang di-export lewat graph-capture (operator fusion ONNX Runtime kurang optimal
> dibanding kernel PyTorch yang sudah dituning). Ini tidak masalah karena MViTv2 memang bukan
> kandidat deployment Android — cuma untuk kelengkapan dokumentasi/pembanding.

## Belum Dilakukan

Integrasi `cnnsa_bilstm.onnx` ke project Android baru (CameraX + MediaPipe Hands on-device +
ONNX Runtime Mobile) — belum ada kodenya, itu langkah berikutnya di luar tahap Export ini.
