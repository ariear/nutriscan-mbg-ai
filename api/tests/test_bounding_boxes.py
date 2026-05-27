"""
Unit tests for InferenceService._extract_bounding_boxes

These tests use synthetic masks (no model or GPU required) to verify
that bounding-box extraction, noise filtering, and JSON serializability
all behave correctly.
"""

import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Make sure `from app.…` imports resolve when running the test directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.inference_service import InferenceService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service() -> InferenceService:
    """Return an InferenceService with dummy dependencies (not used here)."""
    return InferenceService(model=None, fuzzy_clf=None)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_known_regions():
    """
    Build a 256×256 mask with four well-defined class regions and one noise
    blob, then verify that _extract_bounding_boxes returns exactly the
    expected bounding boxes.
    """
    service = _make_service()

    mask = np.zeros((256, 256), dtype=np.int64)

    # Class 1 ("buah"): rows 10–59, cols 20–79  →  50×60 rectangle
    mask[10:60, 20:80] = 1

    # Class 2 ("karbohidrat"): rows 100–199, cols 50–149  →  100×100 square
    mask[100:200, 50:150] = 2

    # Class 3 ("protein"): 3×3 noise blob (9 px² — below min_area 150)
    mask[230:233, 230:233] = 3

    # Class 4 ("sayur"): two disconnected regions
    mask[0:30, 180:220] = 4    # Region A: rows 0–29,  cols 180–219
    mask[200:240, 10:50] = 4   # Region B: rows 200–239, cols 10–49

    detections = service._extract_bounding_boxes(mask, min_area=150)

    # --- Protein noise should be filtered out ---
    labels = [d["label"] for d in detections]
    assert "protein" not in labels, (
        f"Noise blob for 'protein' should be filtered, got: {labels}"
    )

    # --- 4 total detections: buah(1) + karbohidrat(1) + sayur(2) ---
    assert len(detections) == 4, (
        f"Expected 4 detections, got {len(detections)}: {detections}"
    )

    # --- Sorted by (class_id, y, x) ---
    class_ids = [d["class_id"] for d in detections]
    assert class_ids == [1, 2, 4, 4], (
        f"Expected class_id order [1, 2, 4, 4], got {class_ids}"
    )

    # --- Verify "buah" bbox ---
    buah = detections[0]
    assert buah["label"] == "buah"
    assert (buah["x"], buah["y"]) == (20, 10)
    assert (buah["width"], buah["height"]) == (60, 50)

    # --- Verify "karbohidrat" bbox ---
    karbo = detections[1]
    assert karbo["label"] == "karbohidrat"
    assert (karbo["x"], karbo["y"]) == (50, 100)
    assert (karbo["width"], karbo["height"]) == (100, 100)

    # --- Verify "sayur" two regions (sorted by y first) ---
    sayur_a, sayur_b = detections[2], detections[3]
    assert sayur_a["y"] < sayur_b["y"], "sayur regions should be sorted by y"

    # --- All values must be native Python int ---
    for det in detections:
        for key in ("x", "y", "width", "height", "class_id"):
            assert type(det[key]) is int, (
                f"{key} in {det} is {type(det[key])}, expected int"
            )

    # --- Whole list must be JSON-serializable ---
    json_str = json.dumps(detections)
    assert isinstance(json_str, str)

    print("✅ test_known_regions passed!")
    for d in detections:
        print(f"   - {d['label']:15s} cls={d['class_id']}  "
              f"box=({d['x']}, {d['y']}, {d['width']}, {d['height']})")


def test_empty_mask():
    """All-background mask → empty detections list."""
    service = _make_service()
    mask = np.zeros((256, 256), dtype=np.int64)
    detections = service._extract_bounding_boxes(mask)
    assert detections == [], f"Expected [], got {detections}"
    print("✅ test_empty_mask passed!")


def test_single_class_full_image():
    """Entire image is one class → exactly 1 bounding box covering 256×256."""
    service = _make_service()
    mask = np.full((256, 256), fill_value=2, dtype=np.int64)   # all karbohidrat
    detections = service._extract_bounding_boxes(mask, min_area=150)

    assert len(detections) == 1
    d = detections[0]
    assert d["label"] == "karbohidrat"
    assert (d["x"], d["y"]) == (0, 0)
    assert (d["width"], d["height"]) == (256, 256)
    print("✅ test_single_class_full_image passed!")


def test_all_noise():
    """Every class present but only as tiny blobs → all filtered out."""
    service = _make_service()
    mask = np.zeros((256, 256), dtype=np.int64)

    # Scatter 5×5 blobs (25 px² each, all below default min_area=150)
    for cls_id in range(1, 6):
        r, c = cls_id * 40, cls_id * 40
        mask[r:r+5, c:c+5] = cls_id

    detections = service._extract_bounding_boxes(mask)
    assert detections == [], f"All should be filtered, got {detections}"
    print("✅ test_all_noise passed!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_known_regions()
    test_empty_mask()
    test_single_class_full_image()
    test_all_noise()
    print("\n🎉 All tests passed!")
