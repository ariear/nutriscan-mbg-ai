import logging
from pathlib import Path

import tensorflow as tf

from app.core.config import MODEL_PATH

logger = logging.getLogger(__name__)


def load_saved_model(model_path: Path = MODEL_PATH) -> tf.types.experimental.GenericFunction:
    logger.info(f"Loading SavedModel dari: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"SavedModel tidak ditemukan di: {model_path}\n"
            "Pastikan path MODEL_PATH di config.py sudah benar."
        )
    model = tf.saved_model.load(str(model_path))
    logger.info("Model berhasil dimuat.")
    return model
