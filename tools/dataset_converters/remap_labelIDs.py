
#!/usr/bin/env python3
"""
Simple semantic remapper for CropAndWeed-style labelIds.

Given a source mapping scheme (e.g., 'Fine24'), this script reads grayscale
PNG masks from dataset_root/labelIds/<scheme> and writes remapped masks to
dataset_root/labelIds/PhenoID using the following target taxonomy:
  - Soil/Background: 0
  - Sugar beet: 1
  - Weed: 2
  - Ignore: 255

Example policy implemented (Fine24 → PhenoID):
  - id == 1 (Sugar beet)      → 1
  - ids 0..7 except 1         → 255 (ignore)
  - ids 8..23                 → 2 (Weed)
  - everything else           → 0 (Soil)

The source scheme name is a global default but can be overridden via CLI.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple
import cv2
import numpy as np
import importlib.util
from tqdm import tqdm

# Default source scheme; can be overridden with --scheme
MAPPING_SCHEME = 'Fine24'

DATASETS = None  # populated by _load_datasets_mapping()


def _load_datasets_mapping():
    """Dynamically load DATASETS from cropandweed/cnw/utilities/datasets.py"""
    global DATASETS
    if DATASETS is not None:
        return DATASETS
    module_path = Path(__file__).parent / 'cropandweed' / 'cnw' / 'utilities' / 'datasets.py'
    if not module_path.exists():
        raise ImportError(f"Could not locate datasets.py at {module_path}")
    spec = importlib.util.spec_from_file_location("cnw_utilities_datasets", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create spec for cnw_utilities_datasets")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, 'DATASETS'):
        raise ImportError("datasets.py does not define DATASETS")
    DATASETS = mod.DATASETS
    return DATASETS


def build_fine24_to_phenoid_lut() -> np.ndarray:
    """Build a 256-entry LUT for Fine24 → {0,1,2,255} mapping described above."""
    lut = np.zeros((256,), dtype=np.uint8)
    # default already 0 for all
    # Sugar beet stays 1
    lut[1] = 1
    # 0..7 except 1 → 255 (ignore)
    for sid in range(0, 8):
        if sid == 1:
            continue
        lut[sid] = 255
    # 8..23 → Weed (2)
    for sid in range(8, 24):
        lut[sid] = 2
    # everything else remains 0 (Soil)
    return lut


def get_source_and_target_dirs(dataset_root: Path, scheme: str, target_name: str = 'PhenoID') -> Tuple[Path, Path]:
    """Return the source labelIds dir for the scheme and the output PhenoID dir."""
    src = dataset_root / 'labelIds' / scheme
    if not src.exists():
        raise FileNotFoundError(f"Source labelIds directory not found: {src}")
    # Create sibling directory under labelIds
    tgt = dataset_root / 'labelIds' / target_name
    tgt.mkdir(parents=True, exist_ok=True)
    return src, tgt


def iter_label_files(src_dir: Path) -> Iterable[Path]:
    """Yield all labelId PNG files (flat or nested)."""
    # Support both flat and nested structures
    pngs = list(src_dir.rglob('*.png'))
    # Keep a stable order for reproducibility
    pngs.sort()
    return pngs


def remap_and_save(src_path: Path, dst_root: Path, src_root: Path, lut: np.ndarray) -> None:
    """Apply LUT remap to src_path and save to dst_root, preserving relative path."""
    rel = src_path.relative_to(src_root)
    out_path = dst_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {src_path}")

    remapped = lut[mask]
    # Ensure type is uint8
    remapped = remapped.astype(np.uint8, copy=False)
    # Write as grayscale PNG
    ok = cv2.imwrite(str(out_path), remapped)
    if not ok:
        raise RuntimeError(f"Failed to write remapped mask: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Remap semantic labelIds to PhenoID taxonomy (0/1/2/255)')
    p.add_argument('--dataset-root', type=str, required=True, help='Root directory containing labelIds/<scheme>')
    p.add_argument('--scheme', type=str, default=MAPPING_SCHEME, help='Source mapping scheme name (e.g., Fine24)')
    p.add_argument('--target-name', type=str, default='PhenoID', help='Output subfolder name under labelIds')
    p.add_argument('--limit', type=int, default=-1, help='Limit number of files for quick runs (-1 = no limit)')
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    scheme = args.scheme
    target_name = args.target_name

    # Validate scheme exists in DATASETS (informational)
    datasets_map = _load_datasets_mapping()
    if scheme not in datasets_map:
        # Not fatal for I/O, but warn to help catch typos
        print(f"Warning: scheme '{scheme}' not found in DATASETS; proceeding with filesystem lookup only.")

    src_dir, dst_dir = get_source_and_target_dirs(dataset_root, scheme, target_name)
    files = list(iter_label_files(src_dir))
    if args.limit is not None and args.limit > 0:
        files = files[: args.limit]
    print(f"Remapping {len(files)} file(s) from '{src_dir}' → '{dst_dir}' using Fine24→PhenoID policy…")

    lut = build_fine24_to_phenoid_lut()
    errors = 0
    for f in tqdm(files, desc='Remap'):
        try:
            remap_and_save(f, dst_dir, src_dir, lut)
        except Exception as e:
            errors += 1
            print(f"Error processing {f}: {e}")

    print(f"Done. Wrote remapped masks to: {dst_dir}")
    if errors:
        print(f"Completed with {errors} error(s). See logs above.")


if __name__ == '__main__':
    main()

