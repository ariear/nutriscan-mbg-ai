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

    def _extract_bounding_boxes(
        self,
        pred_mask: np.ndarray,
        min_area: int = 150,
    ) -> list[dict]:
        """
        Extract axis-aligned bounding boxes for every detected food class.

        For each non-background class present in `pred_mask`, the method:
          1. Builds a binary mask isolating that class.
          2. Runs cv2.findContours to locate all connected regions.
          3. Filters out noise contours whose area is below `min_area`.
          4. Converts each surviving contour to a bounding rect via
             cv2.boundingRect and records the result.

        Args:
            pred_mask: 2-D NumPy array of shape (H, W) with integer class IDs.
            min_area:  Minimum contour area (in pixels) to keep. Contours
                       smaller than this are treated as segmentation noise
                       and discarded. Default: 150 px².

        Returns:
            A list of dicts ordered by class_id, each with the keys:
              - label     (str)  : Human-readable food class name.
              - class_id  (int)  : Integer class ID from CLASS_MAPPING.
              - x         (int)  : Left edge of the bounding box (pixels).
              - y         (int)  : Top edge of the bounding box (pixels).
              - width     (int)  : Width of the bounding box (pixels).
              - height    (int)  : Height of the bounding box (pixels).
        """
        detections: list[dict] = []

        for cls_id, label in CLASS_MAPPING.items():
            # Skip the background class — it needs no bounding box.
            if cls_id == 0:
                continue

            # Build a uint8 binary mask: 255 where this class was predicted.
            binary_mask = np.where(pred_mask == cls_id, 255, 0).astype(np.uint8)

            # Skip entirely if the class is absent in this prediction.
            if binary_mask.max() == 0:
                continue

            # Find external contours of connected regions.
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    # Ignore tiny blobs that are likely segmentation noise.
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "label"    : label,
                    "class_id" : cls_id,
                    "x"        : int(x),
                    "y"        : int(y),
                    "width"    : int(w),
                    "height"   : int(h),
                })

        # Sort by class_id so the order is deterministic.
        detections.sort(key=lambda d: (d["class_id"], d["y"], d["x"]))
        return detections

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

        foto_ompreng_b64       = base64.b64encode(image_bytes).decode('utf-8')
        segmentasi_makanan_b64 = self._mask_to_base64(pred_mask)
        deteksi_makanan        = self._extract_bounding_boxes(pred_mask)

        return {
            "nutrisi_proporsi"  : nutrisi_prop,
            "status"            : clf["status"],
            "detail"            : clf["detail"],
            "healthy_score"     : clf["healthy_score"],
            "rekomendasi"       : rekomendasi,
            "foto_ompreng"      : foto_ompreng_b64,
            "segmentasi_makanan": segmentasi_makanan_b64,
            "deteksi_makanan"   : deteksi_makanan,
        }
