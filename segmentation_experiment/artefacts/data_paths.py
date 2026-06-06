"""
Shared resolution for on-disk data (same intent as E1-augmentation/training.py paths).

training.py uses DATA_PATH = '../../../../../vast/s222440401' relative to cwd
(typically E1-augmentation/), which resolves to /vast/s222440401.

A naive copy using five .parent steps from this package pointed at /home/vast/...
(one level short of '/'). Use six .parent steps from __file__, or SEGMENTATION_DATA_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path

_ARTEFACTS_DIR = Path(__file__).resolve().parent


def resolve_data_path() -> Path:
    env = os.environ.get("SEGMENTATION_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # .../project/segmentation_experiment/artefacts -> five parents to '/'
    file_to_root = _ARTEFACTS_DIR.parent.parent.parent.parent.parent
    candidate = file_to_root / "vast" / "s222440401"
    if candidate.is_dir():
        return candidate
    fallback = Path("/vast/s222440401")
    if fallback.is_dir():
        return fallback.resolve()
    return candidate


def data_paths():
    root = resolve_data_path()
    return root, root / "training_images", root / "segmentations"
