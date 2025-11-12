#!/usr/bin/env python3
"""
Convert CropAndWeed dataset to Cityscapes format using Datumaro.

This script loads the CropAndWeed dataset (with CSV bboxes and PNG semantic masks)
and converts it to Cityscapes panoptic segmentation format.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import gc
import cv2
import numpy as np
import importlib.util
from tqdm import tqdm
from PIL import Image as PILImage

try:
    import datumaro as dm
    from datumaro.components.annotation import Bbox, Mask, Polygon, Points
    from datumaro.components.media import Image
    from datumaro.components.dataset import Dataset as DatumaroDataset
    from datumaro.components.annotation import LabelCategories, MaskCategories, PointsCategories
except ImportError:
    raise ImportError(
        "Datumaro is not installed. Please install: pip install datumaro"
    )

# Lazy-load DATASETS mapping from the original utilities via file path to avoid import issues


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert CropAndWeed dataset to Cityscapes format using Datumaro'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory of CropAndWeed dataset (containing bboxes, labelIds, images subdirs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for Cityscapes format dataset'
    )
    parser.add_argument(
        '--dataset-variant',
        type=str,
        choices=['CropAndWeed', 'Fine24'],
        default='Fine24',
        help='Dataset variant to use for label mapping'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split name (default: train)'
    )
    parser.add_argument(
        '--use-eval-bboxes',
        action='store_true',
        help='Use Eval bboxes which include vegetation class for unmapped instances'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Limit number of images to process (useful for debugging). Use -1 for no limit.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate matplotlib visualizations of image with ground truth panoptic overlay.'
    )
    parser.add_argument(
        '--viz-limit',
        type=int,
        default=5,
        help='Max number of visualizations to produce (ignored if --visualize not set).'
    )
    parser.add_argument(
        '--viz-output-dir',
        type=str,
        default='viz_debug',
        help='Directory to save visualization PNGs (created if missing).'
    )
    parser.add_argument(
        '--viz-remap-sugarbeet-weeds',
        action='store_true',
        help='Visualize a second panel with remapped IDs: Soil->0, Sugar beet->1, Weeds->3, Others->255.'
    )
    parser.add_argument(
        '--soil-id-in-mask',
        type=int,
        default=24,
        help='ID used for Soil/Background in the raw semantic masks (default: 24).'
    )
    parser.add_argument(
        '--save-remapped-labelids',
        action='store_true',
        help='Save remapped labelIds PNGs (grayscale) under viz_output_dir/remapped_labelIds.'
    )
    
    return parser.parse_args()


def load_cropandweed_dataset(
    dataset_root: str,
    dataset_variant: str,
    use_eval_bboxes: bool = False
) -> Tuple[Dict, List[str]]:
    """
    Load CropAndWeed dataset structure.
    
    Args:
        dataset_root: Root directory containing bboxes, labelIds, images
        dataset_variant: Dataset variant name ('CropAndWeed' or 'Fine24')
        use_eval_bboxes: Whether to use Eval bbox annotations
        
    Returns:
        Tuple of (dataset_config, list of image basenames)
    """
    datasets_map = _load_datasets_mapping()
    if dataset_variant not in datasets_map:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}. Available: {list(DATASETS.keys())}")
    
    dataset_config = datasets_map[dataset_variant]
    
    # Determine bbox directory
    bbox_suffix = 'Eval' if use_eval_bboxes else ''
    bbox_dir = Path(dataset_root) / 'bboxes' / f'{dataset_variant}{bbox_suffix}'
    labelids_dir = Path(dataset_root) / 'labelIds' / dataset_variant
    images_dir = Path(dataset_root) / 'images'
    
    # Check directories exist
    if not bbox_dir.exists():
        raise FileNotFoundError(f"Bboxes directory not found: {bbox_dir}")
    if not labelids_dir.exists():
        raise FileNotFoundError(f"LabelIds directory not found: {labelids_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Get list of CSV files (one per image)
    csv_files = sorted(bbox_dir.glob('*.csv'))
    image_basenames = [f.stem for f in csv_files]
    
    print(f"Found {len(image_basenames)} images in {dataset_variant} variant")
    
    return {
        'config': dataset_config,
        'bbox_dir': bbox_dir,
        'labelids_dir': labelids_dir,
        'images_dir': images_dir,
        'variant': dataset_variant,
    }, image_basenames


def create_datumaro_categories(dataset_config):
    """
    Create Datumaro category definitions from CropAndWeed dataset config.
    
    Args:
        dataset_config: Dataset configuration from DATASETS
        
    Returns:
        List of Datumaro Label categories
    """
    categories = LabelCategories()
    mask_categories = MaskCategories()
    points_categories = PointsCategories()
    
    # Background (Soil) is always stuff class (id=0 in CropAndWeed)
    # For Fine24, there's no explicit soil class, so we add background
    label_ids = sorted(dataset_config.get_label_ids())
    # TODO should I add a soil/background class here with id=0? and Reindex rest of the classes to +1?
    for label_id in label_ids:
        label_name = dataset_config.get_label_name(label_id)
        label_color = dataset_config.get_label_color(label_id, bgr=False)  # RGB
        
        # Soil/Background is stuff, everything else is thing
        # In panoptic segmentation: stuff classes don't have instances
        is_stuff = (label_name == 'Soil' or label_id == 0)
        
        # LabelCategories.add expects a list of attribute names, not values
        # Expose 'is_crowd' as a supported attribute name; values are provided on annotations
        categories.add(
            name=label_name,
            parent='',
            attributes=['is_crowd']
        )
        # Provide colormap for masks as well (use RGB tuple)
        try:
            if label_color is not None and len(label_color) == 3:
                # MaskCategories stores colormap by label index in LabelCategories order,
                # but Datumaro aligns by label id when both categories are provided.
                mask_categories.colormap[len(categories.items) - 1] = tuple(int(c) for c in label_color)
        except Exception:
            pass
        # Provide a single keypoint schema ('stem') for all labels
        try:
            points_categories.add(len(categories.items) - 1, labels=['stem'], joints=[])
        except Exception:
            pass
    
    return categories, mask_categories, points_categories


def load_image_annotations(
    image_basename: str,
    dataset_info: Dict
) -> Tuple[np.ndarray, List[Dict], str]:
    """
    Load annotations for a single image.
    
    Args:
        image_basename: Image name without extension
        dataset_info: Dataset information dict
        
    Returns:
        Tuple of (semantic_mask, bbox_instances, image_path)
    """
    bbox_dir = dataset_info['bbox_dir']
    labelids_dir = dataset_info['labelids_dir']
    images_dir = dataset_info['images_dir']
    dataset_config = dataset_info['config']
    
    # Load semantic mask
    mask_path = labelids_dir / f'{image_basename}.png'
    if mask_path.exists():
        semantic_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        # If no mask exists, create empty one
        semantic_mask = None
    
    # Load bounding boxes from CSV
    bbox_instances = []
    csv_path = bbox_dir / f'{image_basename}.csv'
    
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(
                f,
                fieldnames=['left', 'top', 'right', 'bottom', 'label_id', 'stem_x', 'stem_y']
            )
            for row in reader:
                bbox_instances.append({
                    'bbox': [
                        int(row['left']),
                        int(row['top']),
                        int(row['right']),
                        int(row['bottom'])
                    ],
                    'label_id': int(row['label_id']),
                    'stem': (int(row['stem_x']), int(row['stem_y']))
                })
    
    # Find image file (robust to case and nested dirs)
    def _find_image_file(images_root: Path, stem: str) -> str | None:
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF']
        for ext in exts:
            candidate = images_root / f'{stem}{ext}'
            if candidate.exists():
                return str(candidate)
        # Fallback: recursive search for performance only if direct lookup failed
        try:
            for ext in exts:
                matches = list(images_root.rglob(f'{stem}{ext}'))
                if matches:
                    return str(matches[0])
        except Exception:
            pass
        return None

    image_path = _find_image_file(images_dir, image_basename)
    
    return semantic_mask, bbox_instances, image_path


def load_image_params(dataset_root: str, image_basename: str) -> Dict[str, str]:
    """Load per-image parameters from params CSV and return descriptive strings.

    Supported keys: moisture, soil, lighting, separability
    Value mappings:
      - moisture: 0->dry, 1->medium, 2->wet
      - soil: 0->fine, 1->medium, 2->coarse
      - lighting: 0->sunny, 1->diffuse
      - separability: 0->easy, 1->medium, 2->hard

    CSV formats supported:
      - Headered single-row CSV with those column names
      - Two-column key,value rows without header

    Returns empty dict if file not found or parse fails.
    """
    params_path = Path(dataset_root) / 'params' / f'{image_basename}.csv'
    wanted_keys = {'moisture', 'soil', 'lighting', 'separability'}
    mapping = {
        'moisture': {0: 'dry', 1: 'medium', 2: 'wet'},
        'soil': {0: 'fine', 1: 'medium', 2: 'coarse'},
        'lighting': {0: 'sunny', 1: 'diffuse'},
        'separability': {0: 'easy', 1: 'medium', 2: 'hard'},
    }
    out: Dict[str, str] = {}

    if not params_path.exists():
        return out

    try:
        with open(params_path, 'r', newline='', encoding='utf-8') as f:
            # First, try headered CSV
            pos = f.tell()
            reader = csv.DictReader(f)
            if reader.fieldnames:
                lower_fields = [fn.strip().lower() for fn in reader.fieldnames]
                if any(k in lower_fields for k in wanted_keys):
                    row = next(reader, None)
                    if row is not None:
                        for k in wanted_keys:
                            # Find matching field name (case-insensitive)
                            # If missing, skip
                            try:
                                # get value using original fieldname by index
                                if k in row:
                                    v = row[k]
                                else:
                                    # fallback: search case-insensitive
                                    v = None
                                    for orig in row.keys():
                                        if orig.strip().lower() == k:
                                            v = row[orig]
                                            break
                                if v is not None and v != '':
                                    try:
                                        vi = int(float(v))
                                        out[k] = mapping.get(k, {}).get(vi, str(vi))
                                    except Exception:
                                        # Keep original textual value if not numeric
                                        out[k] = str(v)
                            except Exception:
                                pass
                        return out
            # Fallback: key,value rows
            f.seek(pos)
            reader2 = csv.reader(f)
            for r in reader2:
                if not r:
                    continue
                if len(r) >= 2:
                    key = r[0].strip().lower()
                    if key in wanted_keys:
                        try:
                            vi = int(float(r[1]))
                            out[key] = mapping.get(key, {}).get(vi, str(vi))
                        except Exception:
                            # Keep original textual value if not numeric
                            out[key] = str(r[1])
    except Exception:
        return out

    return out


def convert_to_cityscapes_datumaro(
    dataset_root: str,
    output_dir: str,
    dataset_variant: str,
    split: str = 'train',
    use_eval_bboxes: bool = False,
    limit: int = 10,
    visualize: bool = False,
    viz_limit: int = 5,
    viz_output_dir: str = 'viz_debug',
    viz_remap_sugarbeet_weeds: bool = False,
    soil_id_in_mask: int = 24,
    save_remapped_labelids: bool = False,
):
    """
    Convert CropAndWeed dataset to Cityscapes format using Datumaro.
    
    Args:
        dataset_root: Root directory of CropAndWeed dataset
        output_dir: Output directory for Cityscapes format
        dataset_variant: Dataset variant ('CropAndWeed' or 'Fine24')
        split: Dataset split name
        use_eval_bboxes: Whether to use Eval bboxes
    """
    # Load dataset configuration
    dataset_info, image_basenames = load_cropandweed_dataset(
        dataset_root, dataset_variant, use_eval_bboxes
    )
    # Apply debug limit if requested
    if limit is not None and limit > 0:
        image_basenames = image_basenames[:limit]
    
    # Create categories (labels + mask colormap + points schema)
    categories, mask_categories, points_categories = create_datumaro_categories(dataset_info['config'])
    
    # Streaming generator to reduce memory usage
    print(f"\nConverting {len(image_basenames)} images to Datumaro format (streaming)...")
    viz_count = 0

    def item_generator():
        nonlocal viz_count
        for idx_item, image_basename in enumerate(tqdm(image_basenames)):
            semantic_mask, bbox_instances, image_path = load_image_annotations(
                image_basename, dataset_info
            )
            if image_path is None:
                print(f"Warning: Image not found for {image_basename}, skipping")
                continue

            # Determine image dimensions without fully loading image
            if semantic_mask is not None:
                height, width = semantic_mask.shape[:2]
            else:
                try:
                    with PILImage.open(image_path) as pim:
                        width, height = pim.size
                except Exception:
                    # Fallback to OpenCV if PIL fails
                    img_tmp = cv2.imread(image_path)
                    if img_tmp is None:
                        print(f"Warning: Failed to read image {image_path}, skipping")
                        continue
                    height, width = img_tmp.shape[:2]
                    del img_tmp

            # Build annotations list
            annotations = []

            # Add bounding boxes as thing instances
            for inst_id, bbox_inst in enumerate(bbox_instances):
                label_id = bbox_inst['label_id']
                label_name = dataset_info['config'].get_label_name(label_id)
                if label_name is None or label_name == 'Soil':
                    continue
                # Find category index by name
                cat_idx = None
                for i, cat in enumerate(categories.items):
                    if cat.name == label_name:
                        cat_idx = i
                        break
                if cat_idx is None:
                    continue
                x1, y1, x2, y2 = bbox_inst['bbox']
                annotations.append(
                    Bbox(
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        label=cat_idx,
                        id=inst_id,
                        group=inst_id,
                        attributes={'stem_x': bbox_inst['stem'][0], 'stem_y': bbox_inst['stem'][1]}
                    )
                )

                # Instance mask
                inst_mask = np.zeros((height, width), dtype=np.uint8)
                x1c = max(0, min(width - 1, x1))
                y1c = max(0, min(height - 1, y1))
                x2c = max(0, min(width, x2))
                y2c = max(0, min(height, y2))

                if semantic_mask is not None and x2c > x1c and y2c > y1c:
                    region = (semantic_mask == label_id)
                    win = np.zeros_like(region, dtype=bool)
                    win[y1c:y2c, x1c:x2c] = True
                    guided = region & win
                    if np.any(guided):
                        inst_mask[guided] = 1
                    else:
                        inst_mask[y1c:y2c, x1c:x2c] = 1
                else:
                    if x2c > x1c and y2c > y1c:
                        inst_mask[y1c:y2c, x1c:x2c] = 1

                annotations.append(
                    Mask(
                        image=inst_mask,
                        label=cat_idx,
                        id=inst_id,
                        group=inst_id,
                        attributes={'is_crowd': False}
                    )
                )

                # Add a single keypoint ('stem') as Points annotation
                try:
                    sx, sy = bbox_inst['stem']
                    # Clamp to image bounds
                    sx = float(np.clip(sx, 0, width - 1))
                    sy = float(np.clip(sy, 0, height - 1))
                    annotations.append(
                        Points(
                            points=[sx, sy],
                            visibility=[2],  # 2 = visible
                            label=cat_idx,
                            id=inst_id,
                            group=inst_id,
                        )
                    )
                except Exception:
                    pass

                # Also provide polygon(s) for Cityscapes exporter compatibility
                try:
                    mask_for_cnt = (inst_mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_for_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Fallback to rectangle if no contour found
                    if not contours:
                        rect_pts = [x1c, y1c, x2c, y1c, x2c, y2c, x1c, y2c]
                        annotations.append(
                            Polygon(
                                points=rect_pts,
                                label=cat_idx,
                                id=inst_id,
                                group=inst_id,
                                attributes={'is_crowd': False}
                            )
                        )
                    else:
                        for cnt in contours:
                            if cnt.shape[0] < 3:
                                continue
                            # Approximate to reduce vertices; epsilon as 1% of perimeter
                            peri = cv2.arcLength(cnt, True)
                            epsilon = 0.01 * peri
                            approx = cv2.approxPolyDP(cnt, epsilon, True)
                            if approx.shape[0] < 3:
                                continue
                            pts = approx.reshape(-1, 2).astype(float)
                            # Clamp to image bounds and flatten
                            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
                            poly_pts = pts.flatten().tolist()
                            annotations.append(
                                Polygon(
                                    points=poly_pts,
                                    label=cat_idx,
                                    id=inst_id,
                                    group=inst_id,
                                    attributes={'is_crowd': False}
                                )
                            )
                except Exception:
                    # Be robust: if polygon extraction fails, continue with masks only
                    pass

            # Semantic masks for ALL labels (Soil as stuff with is_crowd, others as thing regions)
            if semantic_mask is not None:
                if not np.issubdtype(semantic_mask.dtype, np.integer):
                    semantic_mask = semantic_mask.astype(np.uint8)
                unique_labels = np.unique(semantic_mask)
                for lid in unique_labels:
                    lid_int = int(lid)
                    label_name = dataset_info['config'].get_label_name(lid_int)
                    if label_name is None:
                        continue
                    cat_idx = None
                    for i, cat in enumerate(categories.items):
                        if cat.name == label_name:
                            cat_idx = i
                            break
                    if cat_idx is None:
                        continue
                    binary_mask = (semantic_mask == lid_int).astype(np.uint8)
                    annotations.append(
                        Mask(
                            image=binary_mask,
                            label=cat_idx,
                            attributes={'is_crowd': (label_name == 'Soil')}
                        )
                    )

                # Optionally save remapped labelIds if requested (debug aid)
                if save_remapped_labelids:
                    try:
                        os.makedirs(os.path.join(viz_output_dir, 'remapped_labelIds'), exist_ok=True)
                        out_lbl_path = os.path.join(viz_output_dir, 'remapped_labelIds', f'{image_basename}.png')
                        cv2.imwrite(out_lbl_path, semantic_mask)
                    except Exception:
                        pass

            # Load optional per-image parameters as item attributes
            item_attrs = load_image_params(dataset_root, image_basename)

            # Create and optionally visualize
            item = dm.DatasetItem(
                id=image_basename,
                subset=split,
                media=Image.from_file(path=image_path),
                annotations=annotations,
                attributes=item_attrs if item_attrs else None
            )

            # Visualization (raw + remapped) lightweight and limited to viz_limit
            if visualize and viz_count < viz_limit:
                import matplotlib.pyplot as plt
                os.makedirs(viz_output_dir, exist_ok=True)
                mask_path = Path(dataset_root) / 'labelIds' / dataset_variant / f'{image_basename}.png'
                sem_mask_v = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
                if sem_mask_v is None:
                    sem_mask_v = np.zeros((height, width), dtype=np.uint8)
                else:
                    if not np.issubdtype(sem_mask_v.dtype, np.integer):
                        sem_mask_v = sem_mask_v.astype(np.uint8)

                datasets_map_v = _load_datasets_mapping()
                ds_cfg_v = datasets_map_v[dataset_variant]
                legend_entries = []
                unique_labels_v = np.unique(sem_mask_v)
                for lid in unique_labels_v:
                    name = ds_cfg_v.get_label_name(int(lid))
                    if name is not None:
                        legend_entries.append((int(lid), name))

                if viz_remap_sugarbeet_weeds:
                    remapped = np.full_like(sem_mask_v, 255, dtype=np.uint8)
                    remapped[sem_mask_v == soil_id_in_mask] = 0
                    sugar_ids = []
                    for lid in ds_cfg_v.get_label_ids():
                        nm = (ds_cfg_v.get_label_name(lid) or '').lower()
                        if 'sugar' in nm and 'beet' in nm:
                            sugar_ids.append(int(lid))
                    if not sugar_ids:
                        for lid in ds_cfg_v.get_label_ids():
                            if (ds_cfg_v.get_label_name(lid) or '').strip() == 'Sugar beet':
                                sugar_ids.append(int(lid))
                    for sid in sugar_ids:
                        remapped[sem_mask_v == sid] = 1
                    crops_core = {'Maize', 'Sugar beet', 'Soy', 'Sunflower', 'Potato', 'Pea', 'Bean', 'Pumpkin'}
                    weed_ids = []
                    for lid in ds_cfg_v.get_label_ids():
                        nm = ds_cfg_v.get_label_name(lid)
                        if nm is None or nm == 'Soil':
                            continue
                        if nm not in crops_core:
                            weed_ids.append(int(lid))
                    for wid in weed_ids:
                        remapped[sem_mask_v == wid] = 2

                # Figure: raw only or raw + remapped
                if viz_remap_sugarbeet_weeds:
                    fig, ax = plt.subplots(1,2, figsize=(10,5))
                else:
                    fig, ax = plt.subplots(1,1, figsize=(5,5))
                    ax = [ax]
                ax[0].imshow(sem_mask_v, cmap='tab20')
                ax[0].set_title('Raw labelIds')
                ax[0].axis('off')
                legend_text = '\n'.join([f"{lid}: {name}" for lid, name in legend_entries])
                if viz_remap_sugarbeet_weeds:
                    ax[1].imshow(remapped, cmap='tab20')
                    ax[1].set_title('Remapped (Soil=0, Sugar=1, Weed=2, Other=255)')
                    ax[1].axis('off')
                    unique_r = np.unique(remapped)
                    counts = {int(v): int((remapped==v).sum()) for v in unique_r}
                    print(f"[viz] {image_basename}: remapped pixel counts -> {counts}")
                fig.suptitle(f"Image: {image_basename} | Variant={dataset_variant}\nRaw Labels: {legend_text}", fontsize=9)
                fig.tight_layout()
                out_path = os.path.join(viz_output_dir, f"{image_basename}_semantic.png")
                fig.savefig(out_path)
                plt.show()
                plt.close(fig)
                viz_count += 1

            # Encourage GC
            if idx_item % 100 == 0:
                gc.collect()

            yield item

    # Materialize items to avoid lazy generator issues and to compute dataset size reliably
    items = list(item_generator())
    if len(items) == 0:
        print("Warning: No items were generated. Check that image paths and CSV/mask stems match.")
    else:
        print(f"Prepared {len(items)} items for export (subset='{split}').")

    # Build a Dataset from materialized items; only provide categories for types that need them
    dataset = DatumaroDataset.from_iterable(
        items,
        categories={
            dm.AnnotationType.label: categories,
            dm.AnnotationType.mask: mask_categories,
            dm.AnnotationType.points: points_categories,
        },
    )

    # Quick sanity: ensure subset has items (helps diagnose empty exports)
    try:
        subset_obj = dataset.get_subset(split)
        # Accessing len may materialize only indices; if it fails, ignore
        subset_len = len(subset_obj)
        if subset_len == 0:
            print(f"Warning: dataset subset '{split}' has 0 items. Check image paths and split name.")
    except Exception:
        pass

    # Export to Cityscapes format
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to Cityscapes format at {output_dir}...")
    dataset.export(
        str(output_path),
        format='datumaro',
        save_media=True,
    )
    
    # print(f"\n✓ Conversion complete! Dataset saved to {output_dir}")
    # print(f"  - Variant: {dataset_variant}")
    # print(f"  - Split: {split}")
    # print(f"  - Images: {len(image_basenames)} (streamed)")
    # print(f"  - Categories: {len(categories.items)}")


def main():
    """Main entry point."""
    args = parse_args()
    
    convert_to_cityscapes_datumaro(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataset_variant=args.dataset_variant,
        split=args.split,
        use_eval_bboxes=args.use_eval_bboxes,
        limit=args.limit,
        visualize=args.visualize,
        viz_limit=args.viz_limit,
        viz_output_dir=args.viz_output_dir,
        viz_remap_sugarbeet_weeds=args.viz_remap_sugarbeet_weeds,
        soil_id_in_mask=args.soil_id_in_mask,
        save_remapped_labelids=args.save_remapped_labelids,
    )


if __name__ == '__main__':
    main()
