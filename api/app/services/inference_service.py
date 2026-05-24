import io
import logging

import cv2
import numpy as np
import tensorflow as tf
import base64
from app.core.config import CLASS_COLORS
from PIL import Image

from app.core.config import (
    CLASS_MAPPING,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUTRISI_MAPPING,
)
from app.services.fuzzy_service import FuzzyNutritionClassifier
from app.services.gemini_service import generate_recommendation


logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        model,
        fuzzy_clf: FuzzyNutritionClassifier,
    ) -> None:
        self.model      = model
        self.fuzzy_clf  = fuzzy_clf
        self.img_size   = (IMG_HEIGHT, IMG_WIDTH)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img     = np.array(pil_img)[:, :, ::-1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, ...]

    def segment(self, img_tensor: np.ndarray) -> np.ndarray:
        infer = self.model.signatures.get("serving_default")
        if infer is not None:
            input_key  = list(infer.structured_input_signature[1].keys())[0]
            output_key = list(infer.structured_outputs.keys())[0]
            tf_input   = tf.constant(img_tensor)
            pred       = infer(**{input_key: tf_input})[output_key]
        else:
            tf_input = tf.constant(img_tensor)
            pred     = self.model(tf_input, training=False)

        mask = tf.argmax(pred[0], axis=-1).numpy()
        return mask


    def compute_nutrisi_proportion(self, pred_mask: np.ndarray) -> dict:
        pixel_per_class: dict[str, int] = {}
        for cls_id, food_name in CLASS_MAPPING.items():
            if cls_id == 0 or food_name not in NUTRISI_MAPPING:
                continue
            pixel_count = int(np.sum(pred_mask == cls_id))
            if pixel_count > 0:
                pixel_per_class[food_name] = pixel_count

        total_food_pixels = sum(pixel_per_class.values())
        if total_food_pixels == 0:
            return {"karbo": 0.0, "protein": 0.0, "serat": 0.0, "susu": 0.0}

        nutrisi_pixels: dict[str, int] = {"karbo": 0, "protein": 0, "serat": 0, "susu": 0}
        for food_name, pixels in pixel_per_class.items():
            nutrisi_cat = NUTRISI_MAPPING[food_name]
            nutrisi_pixels[nutrisi_cat] += pixels

        return {
            k: round(v / total_food_pixels * 100, 1)
            for k, v in nutrisi_pixels.items()
        }

    def _mask_to_base64(self, mask: np.ndarray) -> str:
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            color_mask[mask == cls_id] = color
            
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', color_mask_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, image_bytes: bytes) -> dict:
        img_tensor   = self.preprocess(image_bytes)
        pred_mask    = self.segment(img_tensor)
        nutrisi_prop = self.compute_nutrisi_proportion(pred_mask)
        clf          = self.fuzzy_clf.classify(nutrisi_prop)

        rekomendasi  = generate_recommendation(
            nutrisi_prop=nutrisi_prop,
            status=clf["status"],
            detail=clf["detail"],
            healthy_score=clf["healthy_score"]
        )
        
        foto_ompreng_b64 = base64.b64encode(image_bytes).decode('utf-8')
        segmentasi_makanan_b64 = self._mask_to_base64(pred_mask)

        return {
            "nutrisi_proporsi": nutrisi_prop,
            "status"          : clf["status"],
            "detail"          : clf["detail"],
            "healthy_score"   : clf["healthy_score"],
            "rekomendasi"     : rekomendasi,
            "foto_ompreng"    : foto_ompreng_b64,
            "segmentasi_makanan": segmentasi_makanan_b64,
        }
