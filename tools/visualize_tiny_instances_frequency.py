#!/usr/bin/env python3
"""Visualize tiny crop/weed instances and their frequency-domain patterns.

This script expects a dataset root with at least:
  - RGB images directory
  - semantic labels directory
  - plant instance labels directory

For one selected image, it finds instance IDs with area < max_area, resolves each
instance class from the semantic map (crop/weed), extracts a padded rectangular
crop, and shows:
  1) Full RGB image with tiny-instance boxes.
  2) Per-instance RGB crop and FFT log-magnitude visualization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class TinyInstance:
    instance_id: int
    class_id: int
    class_name: str
    area: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    crop_rgb: np.ndarray
    crop_mask: np.ndarray
    fft_mag: np.ndarray
    dct_mag: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize tiny crop/weed instances in spatial and FFT domain."
    )
    parser.add_argument("root_folder", type=Path, help="Dataset root path")
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default="rgb",
        help="Relative RGB directory under root_folder",
    )
    parser.add_argument(
        "--semantic-dir",
        type=str,
        default="semantics",
        help="Relative semantic label directory under root_folder",
    )
    parser.add_argument(
        "--instance-dir",
        type=str,
        default="plant_instances",
        help="Relative instance label directory under root_folder",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help=(
            "Image file name to inspect (with or without extension). "
            "If omitted, script scans all RGB images and picks the first with tiny instances."
        ),
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=100,
        help="Keep instances with area strictly smaller than this value",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=6,
        help="Extra background pixels around each instance crop",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=24,
        help="Maximum number of tiny instances to display",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help=(
            "Optional image name to start browsing from (with or without extension). "
            "Ignored when --image-name is provided."
        ),
    )
    return parser.parse_args()


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "array" in data:
                arr = data["array"]
            else:
                arr = data[list(data.keys())[0]]
        return np.asarray(arr).squeeze()
    return np.asarray(Image.open(path)).squeeze()


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _normalize_semantics(sem: np.ndarray) -> np.ndarray:
    sem = np.asarray(sem).astype(np.int64)
    # Accept both common encodings:
    #   crop/weed as (1,2) or as (3,4) and map to (1,2)
    if np.any(sem == 3) or np.any(sem == 4):
        sem = sem.copy()
        sem[sem == 3] = 1
        sem[sem == 4] = 2
    return sem


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    return 0.2989 * rgb_f[..., 0] + 0.5870 * rgb_f[..., 1] + 0.1140 * rgb_f[..., 2]


def _fft_log_magnitude(gray_patch: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray_patch)
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    return np.log1p(mag)


def _dct_matrix(n: int) -> np.ndarray:
    # DCT-II orthonormal basis matrix.
    k = np.arange(n, dtype=np.float64)[:, None]
    x = np.arange(n, dtype=np.float64)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * (x + 0.5) * k / n)
    mat[0, :] = np.sqrt(1.0 / n)
    return mat


def _dct2(gray_patch: np.ndarray) -> np.ndarray:
    arr = gray_patch.astype(np.float64, copy=False)
    h, w = arr.shape
    ch = _dct_matrix(h)
    cw = _dct_matrix(w)
    return ch @ arr @ cw.T


def _dct_log_magnitude(gray_patch: np.ndarray) -> np.ndarray:
    coeff = _dct2(gray_patch)
    return np.log1p(np.abs(coeff))


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return x1, y1, x2, y2


def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    height: int,
    width: int,
    pad: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    return x1, y1, x2, y2


def _resolve_class(sem_values: np.ndarray) -> Optional[int]:
    # Ignore background 0 and choose dominant non-zero semantic class.
    vals = sem_values[sem_values > 0]
    if vals.size == 0:
        return None
    classes, counts = np.unique(vals, return_counts=True)
    return int(classes[np.argmax(counts)])


def find_tiny_instances(
    rgb: np.ndarray,
    sem: np.ndarray,
    inst: np.ndarray,
    max_area: int,
    pad: int,
) -> List[TinyInstance]:
    if sem.shape != inst.shape or rgb.shape[:2] != sem.shape:
        raise ValueError(
            f"Shape mismatch: rgb={rgb.shape[:2]}, sem={sem.shape}, inst={inst.shape}"
        )

    class_names = {1: "crop", 2: "weed"}
    h, w = sem.shape
    out: List[TinyInstance] = []

    inst_ids = np.unique(inst)
    for inst_id in inst_ids:
        if inst_id <= 0:
            continue
        mask = inst == inst_id
        area = int(mask.sum())
        if area >= max_area:
            continue

        cls = _resolve_class(sem[mask])
        if cls not in class_names:
            continue

        raw_bbox = _bbox_from_mask(mask)
        x1, y1, x2, y2 = _pad_bbox(raw_bbox, h, w, pad)

        crop_rgb = rgb[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        gray = _to_gray(crop_rgb)
        fft_mag = _fft_log_magnitude(gray)
        dct_mag = _dct_log_magnitude(gray)

        out.append(
            TinyInstance(
                instance_id=int(inst_id),
                class_id=cls,
                class_name=class_names[cls],
                area=area,
                bbox=(x1, y1, x2, y2),
                crop_rgb=crop_rgb,
                crop_mask=crop_mask,
                fft_mag=fft_mag,
                dct_mag=dct_mag,
            )
        )

    return sorted(out, key=lambda x: (x.class_id, x.area, x.instance_id))


def _resolve_image_stem_paths(
    root: Path,
    rgb_dir: Path,
    sem_dir: Path,
    inst_dir: Path,
    image_name: str,
) -> Tuple[Path, Path, Path]:
    stem = Path(image_name).stem
    candidates = []
    for ext in IMG_EXTS:
        candidates.append((rgb_dir / f"{stem}{ext}", sem_dir / f"{stem}{ext}", inst_dir / f"{stem}{ext}"))
    candidates.append((rgb_dir / image_name, sem_dir / image_name, inst_dir / image_name))

    for rgb_path, sem_path, inst_path in candidates:
        if rgb_path.exists() and sem_path.exists() and inst_path.exists():
            return rgb_path, sem_path, inst_path

    raise FileNotFoundError(
        f"Could not find matching RGB/semantic/instance files for '{image_name}' under {root}."
    )


def _iter_rgb_paths(rgb_dir: Path) -> Sequence[Path]:
    out = []
    for p in sorted(rgb_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def _paths_from_rgb(rgb_path: Path, rgb_dir: Path, sem_dir: Path, inst_dir: Path) -> Tuple[Path, Path]:
    rel = rgb_path.relative_to(rgb_dir)
    return sem_dir / rel, inst_dir / rel


def _overlay_tiny_boxes(ax: plt.Axes, rgb: np.ndarray, tiny_instances: List[TinyInstance]) -> None:
    ax.imshow(rgb)
    ax.set_title("Full Image With Tiny Instances (< max_area)")
    ax.axis("off")
    for t in tiny_instances:
        x1, y1, x2, y2 = t.bbox
        color = "lime" if t.class_name == "crop" else "red"
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 2),
            f"{t.class_name} id={t.instance_id} a={t.area}",
            color=color,
            fontsize=7,
            bbox=dict(facecolor="black", alpha=0.4, pad=1),
        )


def _wait_for_next_or_quit(figs: Sequence[plt.Figure]) -> str:
    """Wait for key event and return 'next' or 'quit'.

    Keys:
      - space / escape / enter / right: next image
      - q: quit browser
    """
    state = {"action": None}

    def _on_key(event):
        key = (event.key or "").lower()
        if key in {" ", "space", "escape", "esc", "enter", "right"}:
            state["action"] = "next"
        elif key in {"q"}:
            state["action"] = "quit"
        if state["action"] is not None:
            for f in figs:
                plt.close(f)

    callbacks = []
    for fig in figs:
        cid = fig.canvas.mpl_connect("key_press_event", _on_key)
        callbacks.append((fig, cid))

    while state["action"] is None:
        plt.pause(0.05)

    for fig, cid in callbacks:
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass
    return state["action"]


def visualize_image(
    rgb_path: Path,
    sem_path: Path,
    inst_path: Path,
    max_area: int,
    padding: int,
    max_show: int,
) -> str:
    rgb = _read_rgb(rgb_path)
    sem = _normalize_semantics(_read_array(sem_path))
    inst = _read_array(inst_path).astype(np.int64)

    tiny = find_tiny_instances(rgb=rgb, sem=sem, inst=inst, max_area=max_area, pad=padding)
    if len(tiny) == 0:
        print(f"[SKIP] {rgb_path.name}: no tiny crop/weed instances with area < {max_area}.")
        return "skip"

    tiny = tiny[:max_show]
    print(f"[SHOW] {rgb_path.name}: {len(tiny)} tiny instances")

    plt.ion()

    fig_full, ax_full = plt.subplots(1, 1, figsize=(10, 10))
    _overlay_tiny_boxes(ax_full, rgb, tiny)
    fig_full.suptitle(f"Image: {rgb_path.name}")
    fig_full.tight_layout()

    n = len(tiny)
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(16, max(3, 2.8 * n)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, item in enumerate(tiny):
        ax0, ax1, ax2, ax3 = axes[i]

        ax0.imshow(item.crop_rgb)
        ax0.set_title(f"RGB Crop\n{item.class_name} id={item.instance_id}, area={item.area}")
        ax0.axis("off")

        # Object mask overlay on crop.
        mask_overlay = item.crop_rgb.copy()
        overlay_color = np.array([255, 0, 0], dtype=np.uint8) if item.class_name == "weed" else np.array([0, 255, 0], dtype=np.uint8)
        mask_overlay[item.crop_mask] = (0.5 * mask_overlay[item.crop_mask] + 0.5 * overlay_color).astype(np.uint8)
        ax1.imshow(mask_overlay)
        ax1.set_title("Crop + Instance Mask")
        ax1.axis("off")

        im = ax2.imshow(item.fft_mag, cmap="magma")
        ax2.set_title("FFT Log Magnitude")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        im2 = ax3.imshow(item.dct_mag, cmap="viridis")
        ax3.set_title("DCT Log Magnitude")
        ax3.axis("off")
        fig.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    print("[NAV] Press Space/Esc to continue, or q to quit.")
    action = _wait_for_next_or_quit([fig_full, fig])
    return action


def _find_start_index(rgb_paths: Sequence[Path], start_name: str) -> int:
    target_stem = Path(start_name).stem
    for idx, p in enumerate(rgb_paths):
        if p.stem == target_stem or p.name == start_name:
            return idx
    return 0


def main() -> None:
    args = parse_args()

    root = args.root_folder
    rgb_dir = root / args.rgb_dir
    sem_dir = root / args.semantic_dir
    inst_dir = root / args.instance_dir

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not sem_dir.exists():
        raise FileNotFoundError(f"Semantic directory not found: {sem_dir}")
    if not inst_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {inst_dir}")

    if args.image_name is not None:
        rgb_path, sem_path, inst_path = _resolve_image_stem_paths(
            root=root,
            rgb_dir=rgb_dir,
            sem_dir=sem_dir,
            inst_dir=inst_dir,
            image_name=args.image_name,
        )
        _ = visualize_image(
            rgb_path=rgb_path,
            sem_path=sem_path,
            inst_path=inst_path,
            max_area=args.max_area,
            padding=args.padding,
            max_show=args.max_show,
        )
        return

    # Browse all images with keyboard navigation.
    rgb_paths = list(_iter_rgb_paths(rgb_dir))
    if len(rgb_paths) == 0:
        print(f"No RGB images found under {rgb_dir}.")
        return

    start_idx = 0
    if args.start_from:
        start_idx = _find_start_index(rgb_paths, args.start_from)
        print(f"Starting browser from index {start_idx}: {rgb_paths[start_idx].name}")

    shown_images = 0
    for rgb_path in rgb_paths[start_idx:]:
        sem_path, inst_path = _paths_from_rgb(
            rgb_path=rgb_path,
            rgb_dir=rgb_dir,
            sem_dir=sem_dir,
            inst_dir=inst_dir,
        )
        if not sem_path.exists() or not inst_path.exists():
            continue
        action = visualize_image(
            rgb_path=rgb_path,
            sem_path=sem_path,
            inst_path=inst_path,
            max_area=args.max_area,
            padding=args.padding,
            max_show=args.max_show,
        )
        if action == "skip":
            continue
        shown_images += 1
        if action == "quit":
            print("Stopped by user.")
            return

    if shown_images == 0:
        print(f"No images found with tiny crop/weed instances smaller than {args.max_area} pixels.")
    else:
        print(f"Done. Browsed {shown_images} image(s) with tiny instances.")


if __name__ == "__main__":
    main()
