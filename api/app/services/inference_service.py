"""
InferenceService
Full pipeline: image bytes → preprocessing → segmentation → nutrisi proportion → fuzzy classification.
"""
import io
import logging

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from app.core.config import (
    CLASS_MAPPING,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUTRISI_MAPPING,
)
from app.services.fuzzy_service import FuzzyNutritionClassifier

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Full inference pipeline untuk satu gambar ompreng MBG.
    Model dan fuzzy classifier di-inject saat inisialisasi.
    """

    def __init__(
        self,
        model,
        fuzzy_clf: FuzzyNutritionClassifier,
    ) -> None:
        self.model      = model
        self.fuzzy_clf  = fuzzy_clf
        self.img_size   = (IMG_HEIGHT, IMG_WIDTH)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode image bytes → RGB numpy array → resize → normalize.
        Returns tensor shape (1, H, W, 3) float32 dalam range [0, 1].
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback ke PIL jika cv2 gagal (mis. WebP)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img     = np.array(pil_img)[:, :, ::-1]  # RGB → BGR untuk konsistensi

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, ...]  # (1, H, W, 3)

    # ── Segmentation ──────────────────────────────────────────────────────────

    def segment(self, img_tensor: np.ndarray) -> np.ndarray:
        """
        Jalankan U-Net → softmax output → argmax → 2D mask (H, W).
        SavedModel di-call langsung via __call__ / serving default.
        """
        # SavedModel dari model.export() memiliki signature 'serving_default'
        infer = self.model.signatures.get("serving_default")
        if infer is not None:
            input_key  = list(infer.structured_input_signature[1].keys())[0]
            output_key = list(infer.structured_outputs.keys())[0]
            tf_input   = tf.constant(img_tensor)
            pred       = infer(**{input_key: tf_input})[output_key]
        else:
            # Fallback: callable langsung
            tf_input = tf.constant(img_tensor)
            pred     = self.model(tf_input, training=False)

        # pred shape: (1, H, W, num_classes)
        mask = tf.argmax(pred[0], axis=-1).numpy()  # (H, W)
        return mask

    # ── Nutrisi Proportion ────────────────────────────────────────────────────

    def compute_nutrisi_proportion(self, pred_mask: np.ndarray) -> dict:
        """
        Hitung proporsi (%) tiap kategori nutrisi dari mask prediksi.
        'background' otomatis dilewati karena tidak ada di NUTRISI_MAPPING.
        """
        pixel_per_class: dict[str, int] = {}
        for cls_id, food_name in CLASS_MAPPING.items():
            if cls_id == 0 or food_name not in NUTRISI_MAPPING:
                continue  # skip background
            pixel_count = int(np.sum(pred_mask == cls_id))
            if pixel_count > 0:
                pixel_per_class[food_name] = pixel_count

        total_food_pixels = sum(pixel_per_class.values())
        if total_food_pixels == 0:
            return {"karbo": 0.0, "protein": 0.0, "serat": 0.0, "susu": 0.0}

        # Akumulasi piksel per kategori nutrisi
        nutrisi_pixels: dict[str, int] = {"karbo": 0, "protein": 0, "serat": 0, "susu": 0}
        for food_name, pixels in pixel_per_class.items():
            nutrisi_cat = NUTRISI_MAPPING[food_name]
            nutrisi_pixels[nutrisi_cat] += pixels

        # Proporsi (%)
        return {
            k: round(v / total_food_pixels * 100, 1)
            for k, v in nutrisi_pixels.items()
        }

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes) -> dict:
        """
        Full pipeline: bytes gambar → dict hasil analisis JSON-serializable.

        Returns:
            {
                "nutrisi_proporsi": {"karbo": float, "protein": float, "serat": float, "susu": float},
                "status": str,
                "detail": str,
                "healthy_score": float,
                "rekomendasi": str,
            }
        """
        img_tensor   = self.preprocess(image_bytes)
        pred_mask    = self.segment(img_tensor)
        nutrisi_prop = self.compute_nutrisi_proportion(pred_mask)
        clf          = self.fuzzy_clf.classify(nutrisi_prop)

        rekomendasi  = FuzzyNutritionClassifier.get_rekomendasi(clf["detail"])

        return {
            "nutrisi_proporsi": nutrisi_prop,
            "status"          : clf["status"],
            "detail"          : clf["detail"],
            "healthy_score"   : clf["healthy_score"],
            "rekomendasi"     : rekomendasi,
        }
