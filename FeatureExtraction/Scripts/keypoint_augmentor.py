"""
FeatureExtraction/Scripts/keypoint_augmentor.py
=================================================
Augmentasi sequence keypoint (T, 126) -- dipakai oleh sequence_builder.py.

Kenapa di sini, bukan di PreProcessing (pixel-level)?
Model final (CNNsa-BiLSTM) memakai keypoint MediaPipe, bukan raw pixel frame.
Augmentasi fotometrik (brightness/contrast) tidak berpengaruh ke koordinat
keypoint yang sudah dinormalisasi (relatif ke wrist, scale-invariant) -- jadi
hanya transformasi yang benar-benar mengubah nilai/susunan keypoint yang
berguna sebagai augmentasi: geometris (flip, rotate, zoom), noise deteksi
(jitter), kecepatan gestur (temporal_speed), dan kegagalan deteksi transient
(hand_dropout). Tiga yang terakhir ditambahkan setelah ditemukan gap akurasi
signifikan antara data uji desktop dan pengujian live di aplikasi Android --
tujuannya mensimulasikan variasi kondisi live (signer baru, kamera device
berbeda) dari 5 signer training yang ada, TANPA perlu merekam data baru.

Layout satu baris keypoint (126,): [tangan_kiri(63), tangan_kanan(63)],
tiap tangan = 21 keypoint x (x, y, z) berurutan. Tangan yang tidak
terdeteksi bernilai nol -- semua operasi di bawah ini no-op pada nol.
"""

import numpy as np
from typing import List

KEYPOINTS_PER_HAND = 21
COORDS_PER_KEYPOINT = 3
HAND_BLOCK = KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT  # 63


def apply_horizontal_flip(kpts: np.ndarray) -> np.ndarray:
    """
    Mirror sequence secara horizontal: negasikan koordinat x tiap keypoint,
    lalu tukar blok tangan kiri <-> kanan (mencerminkan seluruh scene).

    Args:
        kpts: (T, 126) float32
    """
    flipped = kpts.copy()
    flipped[:, 0::3] *= -1  # x-coord ada di index kelipatan 3 (0,3,6,... di kedua blok tangan)
    left, right = flipped[:, :HAND_BLOCK].copy(), flipped[:, HAND_BLOCK:].copy()
    flipped[:, :HAND_BLOCK] = right
    flipped[:, HAND_BLOCK:] = left
    return flipped


def _rotate_hand_block(block: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotasi (x, y) tiap keypoint di sekitar wrist (origin, karena sudah dinormalisasi)."""
    T = block.shape[0]
    pts = block.reshape(T, KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT).copy()
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    pts[:, :, 0] = x * cos_a - y * sin_a
    pts[:, :, 1] = x * sin_a + y * cos_a
    return pts.reshape(T, HAND_BLOCK)


def apply_rotation(kpts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotasi kedua tangan di sekitar wrist masing-masing. z tidak berubah."""
    angle_rad = np.deg2rad(angle_deg)
    rotated = kpts.copy()
    rotated[:, :HAND_BLOCK] = _rotate_hand_block(kpts[:, :HAND_BLOCK], angle_rad)
    rotated[:, HAND_BLOCK:] = _rotate_hand_block(kpts[:, HAND_BLOCK:], angle_rad)
    return rotated


def apply_zoom(kpts: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Skala jarak tiap keypoint dari wrist -- variasi ringan untuk regularisasi."""
    return kpts * zoom_factor


def _hand_present_mask(kpts: np.ndarray, block_start: int) -> np.ndarray:
    """(T,) bool -- True di frame mana pun blok tangan [block_start:block_start+HAND_BLOCK]
    TIDAK seluruhnya nol (artinya tangan itu terdeteksi di frame tsb)."""
    block = kpts[:, block_start:block_start + HAND_BLOCK]
    return np.any(block != 0, axis=1)


def apply_landmark_jitter(kpts: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Tambah noise Gaussian kecil ke tiap koordinat -- mensimulasikan noise deteksi MediaPipe
    akibat kondisi kamera/pencahayaan yang berbeda dari dataset training (bukan cuma beda
    signer, device kamera live juga menghasilkan presisi landmark yang beda). Noise ditambah
    SETELAH normalisasi wrist-relative, jadi tidak "dihapus" oleh normalisasi itu.

    Frame/tangan yang memang tidak terdeteksi (blok nol, lihat _hand_present_mask) SENGAJA
    tidak diberi noise -- kalau ikut di-jitter, nilai nol yang berarti "tidak ada tangan" jadi
    angka kecil non-nol yang bisa disalahartikan model sebagai tangan valid yang terdeteksi
    samar-samar.
    """
    jittered = kpts.copy()
    noise = rng.normal(0, noise_std, size=kpts.shape).astype(np.float32)
    for block_start in (0, HAND_BLOCK):
        mask = _hand_present_mask(kpts, block_start)
        jittered[mask, block_start:block_start + HAND_BLOCK] += noise[mask, block_start:block_start + HAND_BLOCK]
    return jittered


def apply_temporal_speed_variation(kpts: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Ubah kecepatan gestur dengan resample sepanjang sumbu waktu (uniform index sampling) --
    speed_factor > 1 mempercepat (jumlah frame berkurang), < 1 memperlambat (jumlah frame
    bertambah). Mensimulasikan variasi kecepatan bergestur antar signer/device, yang sejak
    awal jadi salah satu tersangka utama untuk masalah generalisasi (lihat catatan temporal
    alignment di aplikasi Android). Panjang T hasilnya BOLEH beda dari input -- pad_or_crop_
    sequence() di sequence_builder.py yang menyesuaikan ke SEQUENCE_LENGTH final, jadi
    kompatibel tanpa perlu perubahan di pipeline lain.
    """
    T = kpts.shape[0]
    new_T = max(1, int(round(T / speed_factor)))
    indices = np.linspace(0, T - 1, new_T).round().astype(int)
    return kpts[indices]


def apply_hand_dropout(
    kpts: np.ndarray,
    dropout_prob: float,
    max_frames_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Dengan probabilitas dropout_prob (dicek terpisah per tangan), paksa SEBAGIAN frame
    berturut-turut (durasi acak, maksimum max_frames_frac dari total panjang sequence) pada
    satu tangan jadi nol -- mensimulasikan MediaPipe gagal mendeteksi tangan sesaat (oklusi,
    keluar frame sebentar, gerakan terlalu cepat) seperti yang biasa terjadi di kamera live,
    jauh lebih jarang terjadi di video training yang terkontrol. Sengaja cuma sebagian
    frame (bukan seluruh sequence) supaya realistis sebagai kegagalan TRANSIENT, bukan
    tangan yang memang tidak pernah dipakai di gestur itu.
    """
    dropped = kpts.copy()
    T = kpts.shape[0]
    for block_start in (0, HAND_BLOCK):
        if not _hand_present_mask(kpts, block_start).any():
            continue  # tangan ini memang tidak pernah terdeteksi di sequence asli -- skip
        if rng.random() >= dropout_prob:
            continue
        drop_len = int(rng.integers(1, max(2, int(T * max_frames_frac) + 1)))
        start = int(rng.integers(0, max(1, T - drop_len + 1)))
        dropped[start:start + drop_len, block_start:block_start + HAND_BLOCK] = 0.0
    return dropped


def mix_sequences(kpts_a: np.ndarray, kpts_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Signer-Mixup: interpolasi linear antara 2 sequence keypoint dari KELAS YANG SAMA tapi
    SIGNER BERBEDA, menghasilkan "signer sintetis" yang posisinya di ANTARA keduanya di ruang
    fitur. Beda secara mendasar dari augmentasi lain di atas (flip/rotate/zoom/jitter/dst) yang
    cuma memvariasikan SATU signer (menggoyang 1 titik data) -- ini mengisi CELAH antar signer
    asli, memperluas keragaman signer yang benar-benar dilihat model, bukan cuma memperbanyak
    variasi di sekitar 5 titik yang ada. Dipanggil dari sequence_builder.py (perlu tahu signer
    tiap sample, augment_sequence() di atas sengaja tidak, karena cuma butuh 1 sequence).

    kpts_b diresample dulu ke panjang kpts_a (nearest-index via linspace, pola sama seperti
    KeypointSequenceBuilder.kt di app Android) supaya bisa dicampur elementwise walau video
    sumbernya beda durasi asli.

    Args:
        kpts_a, kpts_b: (T_a, 126) dan (T_b, 126) float32 -- KELAS SAMA, SIGNER BEDA.
        alpha: bobot campuran (0-1). Sebaiknya dijaga caller di rentang tidak ekstrem (mis.
               0.3-0.7) supaya hasilnya benar-benar "di antara" keduanya, bukan sekadar salinan
               nyaris murni salah satu induk.

    Returns:
        (T_a, 126) float32 -- panjang sequence mengikuti kpts_a.
    """
    T_a = kpts_a.shape[0]
    T_b = kpts_b.shape[0]
    if T_b != T_a:
        indices = np.linspace(0, T_b - 1, T_a).round().astype(int)
        kpts_b = kpts_b[indices]
    return (alpha * kpts_a + (1.0 - alpha) * kpts_b).astype(np.float32)


def augment_sequence(
    kpts: np.ndarray,
    techniques: List[str],
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Terapkan augmentasi random ke satu sequence keypoint. Urutan: transformasi spasial dulu
    (flip/rotate/zoom/jitter, tidak bergantung panjang T) -> temporal_speed (baru mengubah T)
    -> hand_dropout terakhir (durasi drop dihitung dari T final setelah speed variation).

    Args:
        kpts       : (T, 126) float32
        techniques : list teknik yang diaktifkan
        config     : AUGMENTATION_CONFIG dari feature_config.py
        rng        : numpy random generator (untuk reproducibility)

    Returns:
        np.ndarray (T', 126) yang sudah di-augment -- T' == T kecuali "temporal_speed" aktif.
    """
    aug = kpts.copy()

    if "horizontal_flip" in techniques:
        if rng.random() < config.get("horizontal_flip_prob", 0.5):
            aug = apply_horizontal_flip(aug)

    if "rotation" in techniques:
        lo, hi = config.get("rotation_range", (-10, 10))
        aug = apply_rotation(aug, rng.uniform(lo, hi))

    if "zoom" in techniques:
        lo, hi = config.get("zoom_range", (0.9, 1.1))
        aug = apply_zoom(aug, rng.uniform(lo, hi))

    if "jitter" in techniques:
        noise_std = config.get("jitter_std", 0.02)
        aug = apply_landmark_jitter(aug, noise_std, rng)

    if "temporal_speed" in techniques:
        lo, hi = config.get("temporal_speed_range", (0.85, 1.15))
        aug = apply_temporal_speed_variation(aug, rng.uniform(lo, hi))

    if "hand_dropout" in techniques:
        prob = config.get("hand_dropout_prob", 0.15)
        max_frac = config.get("hand_dropout_max_frac", 0.3)
        aug = apply_hand_dropout(aug, prob, max_frac, rng)

    return aug
