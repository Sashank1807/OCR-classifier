import pytest
import numpy as np
import cv2
from pathlib import Path
from app.services.preprocessor import preprocessor


def test_perspective_homography_warp():
    # Create a 400x300 image with a white rectangle on a black background
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 250), (255, 255, 255), -1)

    # Distorted quad corners [top-left, top-right, bottom-right, bottom-left]
    quad = np.array([
        [40.0, 60.0],
        [360.0, 40.0],
        [340.0, 260.0],
        [60.0, 240.0]
    ], dtype=np.float32)

    warped = preprocessor.warp_perspective(img, quad)

    assert warped is not None
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0
    # Output should be rectangular
    assert isinstance(warped, np.ndarray)


def test_process_table_preserves_decimals():
    # Create an image containing a thin decimal point
    img = np.full((100, 200, 3), 245, dtype=np.uint8)
    # Draw dark text with decimal: "12.50"
    cv2.putText(img, "12.50", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    processed = preprocessor.process_table(img)
    assert processed is not None
    assert processed.shape == img.shape
    # Check that decimal region is not washed out
    decimal_region = processed[50:65, 70:85]
    assert np.min(decimal_region) < 100  # Dark pixel preserved
