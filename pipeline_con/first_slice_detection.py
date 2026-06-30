import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

YOLO_IMGSZ = 512
YOLO_CONF = 0.70
YOLO_MAX_GAP = 3
YOLO_BATCH = 32
yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'

HU_MIN, HU_MAX = -200, 1800

def slice_to_image(slice_2d: np.ndarray) -> np.ndarray:
    """Window-normalise an axial HU slice to an imgsz 3-channel uint8 image."""
    windowed = np.clip(slice_2d, HU_MIN, HU_MAX)
    scaled = (windowed - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
    img = Image.fromarray(scaled.astype(np.uint8)).resize((YOLO_IMGSZ, YOLO_IMGSZ))
    gray = np.asarray(img, dtype=np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)

def longest_present_run(present: np.ndarray, max_gap: int = YOLO_MAX_GAP):
    """Longest run of True values, tolerating up to ``max_gap`` gaps.

    Returns (start_index, length); start ignores leading gap slices so it
    points at a real detection. (-1, 0) if there is no detection at all.
    """
    best_start, best_len = -1, 0
    run_start, run_present, gap = -1, 0, 0
    for i, val in enumerate(present):
        if val:
            if run_start == -1:
                run_start = i
            run_present += 1
            gap = 0
        elif run_start != -1:
            gap += 1
            if gap > max_gap:
                if run_present > best_len:
                    best_start, best_len = run_start, run_present
                run_start, run_present, gap = -1, 0, 0
    if run_present > best_len:
        best_start, best_len = run_start, run_present
    return best_start, best_len


def find_first_slice(yolo_model: YOLO, volume: np.ndarray) -> int:
    """Run YOLO over every axial slice and return the first vertebra slice index.

    Mirrors first_slice_detection/find_first_slice.py: scans down the z axis for
    the longest run of slices with a detection above YOLO_CONF and returns the
    first slice of that run (-1 if no vertebra is detected).
    """
    num_slices = volume.shape[0]
    confs = np.zeros(num_slices, dtype=np.float32)
    for start in range(0, num_slices, YOLO_BATCH):
        stop = min(start + YOLO_BATCH, num_slices)
        batch = [slice_to_image(volume[z]) for z in range(start, stop)]
        results = yolo_model.predict(
            batch,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            device=yolo_device,
            verbose=False,
        )
        for offset, res in enumerate(results):
            if res.boxes is not None and len(res.boxes) > 0:
                confs[start + offset] = float(res.boxes.conf.max())
    present = confs >= YOLO_CONF
    start, _ = longest_present_run(present)
    return start

