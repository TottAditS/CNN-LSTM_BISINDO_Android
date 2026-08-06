"""
FeatureExtraction/Config/feature_config.py
Auto-detect ROOT dan konfigurasi semua setting untuk feature extraction.
"""

from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Datasets").exists() and (parent / "PreProcessing").exists():
            return parent
    raise RuntimeError(
        "Tidak bisa menemukan root project. "
        "Pastikan direktori Datasets/ dan PreProcessing/ ada di root."
    )


# --- ROOT & PATH --------------------------------------------------------------
ROOT = find_project_root()

DATASET_DIR         = ROOT / "Datasets" / "WLBisindo"
PROCESSED_DIR       = ROOT / "Datasets" / "WLBisindo_Processed"
FEATURE_OUTPUT_DIR  = ROOT / "Datasets" / "WLBisindo_Features"

KEYPOINT_DIR        = FEATURE_OUTPUT_DIR / "keypoints"    # per-video .npy
SEQUENCE_DIR        = FEATURE_OUTPUT_DIR / "sequences"    # final .npz


# --- DATASET INFO -------------------------------------------------------------
# Sesuai isi Datasets/WLBisindo saat ini: signer0-signer4, label0-label31.
# signer0 baru merekam 12/32 label; sample per (signer,label) bervariasi 10 atau 15.
# Update nilai ini jika dataset bertambah (mis. lengkap jadi 40 label).
NUM_SIGNERS                  = 5
NUM_LABELS                   = 32
SAMPLES_PER_LABEL_PER_SIGNER = None  # bervariasi, tidak dipakai untuk enumerasi file


# --- FRAME SETTINGS (harus sama dengan PreProcessing) -------------------------
FRAME_SIZE       = (224, 224)
MAX_FRAMES       = 60
SEQUENCE_LENGTH  = 30


# --- MEDIAPIPE SETTINGS -------------------------------------------------------
NUM_HANDS           = 2
KEYPOINTS_PER_HAND  = 21
COORDS_PER_KEYPOINT = 3
FEATURES_PER_FRAME  = NUM_HANDS * KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT  # 126

MP_STATIC_IMAGE_MODE        = False
MP_MAX_NUM_HANDS            = NUM_HANDS
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE  = 0.5


# --- NORMALISASI KEYPOINT -----------------------------------------------------
# Subtract wrist, scale by wrist->MCP9 distance -> invariant terhadap posisi & ukuran tangan
NORMALIZE_KEYPOINTS = True


# --- SEQUENCE SETTINGS --------------------------------------------------------
PAD_VALUE            = 0.0
CENTER_CROP_TEMPORAL = True   # True = ambil frame tengah; False = ambil N frame pertama


# --- SPLIT SETTINGS -----------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


# --- AUGMENTATION SETTINGS (keypoint-level) ------------------------------------
# Augmentasi dilakukan di sini (bukan di PreProcessing) karena model final
# memakai keypoint MediaPipe (T, 126), bukan raw pixel frame -- augmentasi
# fotometrik (brightness/contrast) tidak berpengaruh ke koordinat keypoint
# yang sudah dinormalisasi, jadi hanya transformasi geometris yang dipakai.
# Augmentasi HANYA diterapkan ke video yang masuk train split (split dilakukan
# dulu sebelum augmentasi, supaya tidak ada data leakage ke val/test).
#
# Range diperlebar + 3 teknik baru (jitter, temporal_speed, hand_dropout) ditambah
# setelah ditemukan gap akurasi besar antara data uji desktop (94,53%) dan pengujian
# live di aplikasi Android (CNNsa-BiLSTM sering salah, walau confidence tinggi).
# Dataset cuma 5 signer -- daripada merekam signer baru (tidak scalable untuk aplikasi
# nyata, lihat diskusi di TugasSaya/rencana_fine_tune_signer_adaptation.md), augmentasi
# yang lebih kuat mensimulasikan variasi signer/kondisi kamera baru dari data yang ada.
AUGMENTATION_CONFIG = {
    "enabled"               : True,
    "techniques"            : ["horizontal_flip", "rotation", "zoom", "jitter", "temporal_speed", "hand_dropout"],
    "horizontal_flip_prob"  : 0.5,
    "rotation_range"        : (-20, 20),    # derajat -- diperlebar dari (-10,10)
    "zoom_range"            : (0.8, 1.2),   # skala jarak tiap keypoint dari wrist -- diperlebar dari (0.9,1.1)
    "jitter_std"            : 0.02,         # std noise Gaussian per koordinat -- simulasi noise deteksi MediaPipe
    "temporal_speed_range"  : (0.85, 1.15), # simulasi variasi kecepatan bergestur antar signer/device
    "hand_dropout_prob"     : 0.15,         # peluang satu tangan "hilang" sebagian frame (per tangan, per sequence)
    "hand_dropout_max_frac" : 0.3,          # maksimum proporsi sequence yang boleh kena dropout sekali kena
    "augment_multiplier"    : 4,            # naik dari 2 -> 1 original + 4 augmented = 5x data train

    # Signer-Mixup: BEDA dari 6 teknik di atas (yang cuma memvariasikan 1 signer) -- ini
    # mencampur (interpolasi linear) sequence dari 2 SIGNER BERBEDA pada kelas yang sama,
    # menghasilkan "signer sintetis" yang mengisi celah antar 5 signer asli. Ditambah setelah
    # augmentasi biasa terbukti menaikkan akurasi test-set (94,53%->95,70%) TAPI tidak menutup
    # gap ke signer baru saat live test di HP -- lihat diskusi & PROGRESS.md repo Android.
    "signer_mixup_count"      : 2,          # jumlah versi mixup tambahan per video train
    "signer_mixup_alpha_range": (0.3, 0.7), # bobot campuran -- dijaga tidak ekstrem dekat 0/1
}


# --- PROCESSING SETTINGS ------------------------------------------------------
VERBOSE = True


# --- LABEL MAP ----------------------------------------------------------------
LABEL_MAP = {i: f"label_{i}" for i in range(NUM_LABELS)}


# --- HELPERS ------------------------------------------------------------------
def get_keypoint_path(signer_id: int, label_id: int, sample_id: int) -> Path:
    fname = f"signer{signer_id}_label{label_id}_sample{sample_id}.npy"
    return KEYPOINT_DIR / fname


def get_sequence_output_path(split: str) -> Path:
    return SEQUENCE_DIR / f"{split}_sequences.npz"


def create_output_dirs():
    for d in [FEATURE_OUTPUT_DIR, KEYPOINT_DIR, SEQUENCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if VERBOSE:
        print(f"[Config] ROOT          : {ROOT}")
        print(f"[Config] Dataset       : {DATASET_DIR}")
        print(f"[Config] Keypoint Dir  : {KEYPOINT_DIR}")
        print(f"[Config] Sequence Dir  : {SEQUENCE_DIR}")
        print(f"[Config] Features/frame: {FEATURES_PER_FRAME} "
              f"({NUM_HANDS} hands x {KEYPOINTS_PER_HAND} kpts x {COORDS_PER_KEYPOINT} coords)")


if __name__ == "__main__":
    create_output_dirs()
    print("Config OK.")
