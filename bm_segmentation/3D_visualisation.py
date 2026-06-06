"""
Render BM training volumes to two static PNG images.

Uses PyVista for surface extraction (marching cubes) and matplotlib for
off-screen rendering — no OpenGL display required.

For a random (or specified) study, saves:
  1. 3D render of bm_images_nii (masked CT isosurface)
  2. 3D render of bm_segmentations_nii (per-label vertebra meshes)
"""

from __future__ import annotations

import argparse
import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")

DATA_PATH = "../../../../../vast/s222440401"
BM_IMAGES_PATH = os.path.join(DATA_PATH, "bm_data/bm_images_nii")
BM_SEG_PATH = os.path.join(DATA_PATH, "bm_data/bm_segmentations_nii")
OUTPUT_DIR = "./vis"

HU_MIN, HU_MAX = -200.0, 1800.0
CT_ISO_THRESHOLD = 1.0  # bm_images background is 0; capture masked region
MESH_DECIMATION = 0.85
CAMERA_ELEV = 25
CAMERA_AZIM = 135

LABEL_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
]


def list_studies() -> list[str]:
    """Return study IDs that have both image and segmentation volumes."""
    image_ids = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(BM_IMAGES_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    seg_ids = {
        f.replace(".nii.gz", "").replace(".nii", "")
        for f in os.listdir(BM_SEG_PATH)
        if f.endswith((".nii", ".nii.gz"))
    }
    return sorted(image_ids & seg_ids)


def pick_random_study(seed: int | None = None) -> str:
    studies = list_studies()
    if not studies:
        raise FileNotFoundError(
            f"No overlapping studies found in {BM_IMAGES_PATH} and {BM_SEG_PATH}"
        )
    rng = random.Random(seed)
    return rng.choice(studies)


def load_nifti(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(path)
    data = np.ascontiguousarray(nii.get_fdata(), dtype=np.float32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    return data, spacing


def numpy_to_image_data(
    volume: np.ndarray, spacing: tuple[float, float, float]
) -> pv.ImageData:
    """Convert a 3D numpy array to a PyVista ImageData grid."""
    grid = pv.ImageData()
    grid.dimensions = volume.shape
    grid.spacing = spacing
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["values"] = volume.ravel(order="F")
    return grid


def _decimate_mesh(mesh: pv.PolyData) -> pv.PolyData:
    if MESH_DECIMATION > 0 and mesh.n_points > 0:
        return mesh.decimate_pro(MESH_DECIMATION)
    return mesh


def volume_isosurface_mesh(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    threshold: float,
) -> pv.PolyData | None:
    """Extract a decimated isosurface from a scalar volume."""
    if not (volume > threshold).any():
        return None

    grid = numpy_to_image_data(volume, spacing)
    mesh = grid.contour([threshold], scalars="values")
    if mesh.n_points == 0:
        return None
    return _decimate_mesh(mesh)


def label_surface_mesh(
    segmentation: np.ndarray,
    label: int,
    spacing: tuple[float, float, float],
) -> pv.PolyData | None:
    """Extract a decimated isosurface mesh for a single segmentation label."""
    mask = (segmentation == label).astype(np.float32)
    if not mask.any():
        return None

    grid = numpy_to_image_data(mask, spacing)
    mesh = grid.contour([0.5], scalars="values")
    if mesh.n_points == 0:
        return None
    return _decimate_mesh(mesh)


def _set_3d_limits(ax, points: np.ndarray) -> None:
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())


def _style_3d_axes(ax, title: str) -> None:
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)


def _add_colored_mesh(ax, mesh: pv.PolyData, color: str) -> None:
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    verts = mesh.points
    collection = Poly3DCollection(
        verts[faces],
        alpha=0.75,
        facecolor=color,
        edgecolor="none",
    )
    ax.add_collection3d(collection)


def _add_hu_colored_mesh(ax, mesh: pv.PolyData) -> None:
    """Render a mesh with per-face colours mapped from HU scalars."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    verts = mesh.points
    scalars = mesh.point_data["values"]
    norm = Normalize(vmin=HU_MIN, vmax=HU_MAX)
    cmap = cm.get_cmap("gray")

    face_colors = []
    for face in faces:
        hu_mean = float(np.mean(scalars[face]))
        face_colors.append(cmap(norm(hu_mean)))

    collection = Poly3DCollection(
        verts[faces],
        alpha=0.85,
        facecolors=face_colors,
        edgecolor="none",
    )
    ax.add_collection3d(collection)


def _save_3d_figure(
    ax,
    title: str,
    output_path: str,
    dpi: int,
    suptitle: str,
) -> None:
    _style_3d_axes(ax, title)
    fig = ax.figure
    fig.suptitle(suptitle, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_bm_image_3d(
    study_id: str,
    output_path: str,
    dpi: int = 150,
) -> str:
    """Render a 3D isosurface of the masked CT volume (bm_images_nii)."""
    image_path = os.path.join(BM_IMAGES_PATH, f"{study_id}.nii")
    image, spacing = load_nifti(image_path)

    mesh = volume_isosurface_mesh(image, spacing, CT_ISO_THRESHOLD)
    if mesh is None:
        raise ValueError(f"No nonzero CT voxels found for study {study_id}")

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _add_hu_colored_mesh(ax, mesh)
    _set_3d_limits(ax, mesh.points)

    _save_3d_figure(
        ax,
        title="3D bm_images_nii (masked CT)",
        output_path=output_path,
        dpi=dpi,
        suptitle=f"BM Image 3D — {study_id}  |  Shape: {image.shape}",
    )
    return output_path


def render_bm_segmentation_3d(
    study_id: str,
    output_path: str,
    dpi: int = 150,
) -> str:
    """Render a 3D mesh per segmentation label (bm_segmentations_nii)."""
    seg_path = os.path.join(BM_SEG_PATH, f"{study_id}.nii")
    segmentation, spacing = load_nifti(seg_path)
    segmentation = segmentation.astype(np.int32)
    labels = sorted(int(v) for v in np.unique(segmentation) if v > 0)

    if not labels:
        raise ValueError(f"No segmentation labels found for study {study_id}")

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    all_points: list[np.ndarray] = []

    for idx, label in enumerate(labels):
        mesh = label_surface_mesh(segmentation, label, spacing)
        if mesh is None:
            continue
        color = LABEL_COLORS[idx % len(LABEL_COLORS)]
        _add_colored_mesh(ax, mesh, color)
        all_points.append(mesh.points)

    if not all_points:
        raise ValueError(f"Could not extract segmentation meshes for study {study_id}")

    _set_3d_limits(ax, np.vstack(all_points))

    _save_3d_figure(
        ax,
        title="3D bm_segmentations_nii",
        output_path=output_path,
        dpi=dpi,
        suptitle=f"BM Segmentation 3D — {study_id}  |  Labels: {labels}",
    )
    return output_path


def render_visualization_images(
    study_id: str,
    output_dir: str | None = None,
    dpi: int = 150,
) -> tuple[str, str]:
    """
    Render both 3D visualisations and save them as PNG files.

    Returns:
        (image_png_path, segmentation_png_path)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    image_out = os.path.join(output_dir, f"{study_id}_bm_images.png")
    seg_out = os.path.join(output_dir, f"{study_id}_bm_segmentations.png")

    render_bm_image_3d(study_id, image_out, dpi=dpi)
    render_bm_segmentation_3d(study_id, seg_out, dpi=dpi)

    return image_out, seg_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render BM 3D visualisations to two PNG images"
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Study ID to visualise (default: random study)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed when choosing a study",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./vis)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image resolution (default: 150)",
    )
    return parser.parse_args()


def main() -> tuple[str, str]:
    args = parse_args()
    study_id = args.study or pick_random_study(seed=args.seed)

    print("=" * 60)
    print("BM 3D Visualisation (PNG export)")
    print("=" * 60)
    print(f"Study:          {study_id}")
    print(f"Images:         {os.path.abspath(BM_IMAGES_PATH)}")
    print(f"Segmentations:  {os.path.abspath(BM_SEG_PATH)}")
    print("=" * 60)

    image_path, seg_path = render_visualization_images(
        study_id,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print(f"Saved bm_images:        {os.path.abspath(image_path)}")
    print(f"Saved bm_segmentations: {os.path.abspath(seg_path)}")
    return image_path, seg_path


if __name__ == "__main__":
    main()
