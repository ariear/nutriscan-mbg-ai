import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR         = Path(__file__).resolve().parents[2]
MODEL_PATH       = BASE_DIR / "models" / "unet_mbg_savedmodel"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

IMG_HEIGHT   = 256
IMG_WIDTH    = 256
IMG_CHANNELS = 3

CLASS_MAPPING: dict[int, str] = {
    0: "background",
    1: "buah",
    2: "karbohidrat",
    3: "protein",
    4: "sayur",
    5: "susu",
}

NUM_CLASSES = len(CLASS_MAPPING)

CLASS_COLORS: dict[int, list[int]] = {
    0: [0,   0,   0  ],
    1: [255, 140, 0  ],
    2: [240, 220, 80 ],
    3: [200, 80,  80 ],
    4: [50,  200, 50 ],
    5: [100, 190, 255],
}

NUTRISI_MAPPING: dict[str, str] = {
    "buah"        : "serat",
    "karbohidrat" : "karbo",
    "protein"     : "protein",
    "sayur"       : "serat",
    "susu"        : "susu",
}

IDEAL_PROPORSI: dict[str, float] = {
    "karbo"  : 50.0,
    "protein": 25.0,
    "serat"  : 15.0,
    "susu"   : 10.0,
}
