"""
Konstanta dan konfigurasi global untuk MBG Inference API.
Diambil dari notebook: deteksi_risiko_pola_makan_mbg_roboflow.ipynb
"""
from pathlib import Path

# ── Path ────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parents[3]  # /deteksi_mbg/
MODEL_PATH       = BASE_DIR / "models" / "unet_mbg_savedmodel"

# ── Image config ─────────────────────────────────────────────────────────────
IMG_HEIGHT   = 256
IMG_WIDTH    = 256
IMG_CHANNELS = 3

# ── Class mapping (dari dataset Roboflow) ────────────────────────────────────
# 0: background, 1: buah, 2: karbohidrat, 3: protein, 4: sayur, 5: susu
# Catatan: 'ompreng' (tray/wadah) sudah dikeluarkan dari dataset label
CLASS_MAPPING: dict[int, str] = {
    0: "background",
    1: "buah",
    2: "karbohidrat",
    3: "protein",
    4: "sayur",
    5: "susu",
}

NUM_CLASSES = len(CLASS_MAPPING)

# ── Warna visualisasi per kelas (RGB) ────────────────────────────────────────
CLASS_COLORS: dict[int, list[int]] = {
    0: [0,   0,   0  ],   # background  → hitam
    1: [255, 140, 0  ],   # buah        → oranye
    2: [240, 220, 80 ],   # karbohidrat → kuning
    3: [200, 80,  80 ],   # protein     → merah
    4: [50,  200, 50 ],   # sayur       → hijau
    5: [100, 190, 255],   # susu        → biru muda
}

# ── Mapping kelas makanan → kategori nutrisi ─────────────────────────────────
NUTRISI_MAPPING: dict[str, str] = {
    "buah"        : "serat",    # buah-buahan      → serat & vitamin
    "karbohidrat" : "karbo",    # nasi/mie/roti dll → karbohidrat
    "protein"     : "protein",  # lauk hewani/nabati→ protein
    "sayur"       : "serat",    # sayuran           → serat
    "susu"        : "susu",     # susu              → kategori tersendiri MBG
}

# ── Proporsi ideal MBG (%) ───────────────────────────────────────────────────
IDEAL_PROPORSI: dict[str, float] = {
    "karbo"  : 50.0,
    "protein": 25.0,
    "serat"  : 15.0,
    "susu"   : 10.0,
}
