"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Ekin Celikkan <ekin.celikkan@gfz-potsdam.de>
# SPDX-License-Identifier: Apache-2.0

Converter script to transform a WeedsGalore-style dataset into an RGB+label dataset
with train/val/test splits suitable for Cityscapes-like consumption.

Usage examples:
  python weedsgalore2rgb.py \
    --src /path/to/weeds_galore_root \
    --dst /path/to/output_root \
    --splits-dir /path/with/split_txts \
    --in-bands 5

By default the script will create the following layout under `--dst`:
  images/train/, images/val/, images/test/
  labels/train/, labels/val/, labels/test/
and write `train.txt`, `val.txt`, `test.txt` (lists of image ids) into `--dst`.

If you want a Cityscapes-style layout, use `--cityscapes-structure`.

The script attempts to reuse the same file layout assumptions as the loader
you provided: band PNGs are named `<image_id>_R.png`, `_G.png`, `_B.png`,
`_NIR.png`, `_RE.png` and are found under `<src>/<image_id[:10]>/images/`.
Semantic label PNGs are under `<src>/<image_id[:10]>/semantics/<image_id>.png`.

"""
from PIL import Image
import argparse
import os
import json
import numpy as np
from pathlib import Path


def to_uint8(arr):
    """Convert array to uint8 [0..255]. Handles floats in [0..1] or ints in larger ranges."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        # assume 0..1
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).round().astype(np.uint8)
    # other integer types, rescale if needed
    maxv = arr.max() if arr.size else 255
    if maxv > 255:
        arr = (arr.astype(np.float32) / maxv) * 255.0
        return arr.round().astype(np.uint8)
    return arr.astype(np.uint8)


def read_band(band_path):
    if not os.path.exists(band_path):
        raise FileNotFoundError(band_path)
    with Image.open(band_path) as im:
        return np.array(im)


def build_rgb(src_root, image_id, in_bands=3):
    """Read bands for a given image_id and build an HxWx3 RGB array."""
    base_dir = os.path.join(src_root, image_id[:10])
    images_dir = os.path.join(base_dir, 'images')
    r = read_band(os.path.join(images_dir, image_id + '_R.png'))
    g = read_band(os.path.join(images_dir, image_id + '_G.png'))
    b = read_band(os.path.join(images_dir, image_id + '_B.png'))
    r = to_uint8(r)
    g = to_uint8(g)
    b = to_uint8(b)
    # Ensure shapes match
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError(f"Band shapes differ for {image_id}: {r.shape}, {g.shape}, {b.shape}")
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def read_label(src_root, image_id):
    base_dir = os.path.join(src_root, image_id[:10])
    sem_dir = os.path.join(base_dir, 'semantics')
    label_path = os.path.join(sem_dir, image_id + '.png')
    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)
    with Image.open(label_path) as im:
        return np.array(im)


def remap_label(label, mapping=None):
    """Remap label values according to mapping dict (keys/values are ints).
    If mapping is None, use default binary mapping: values > 1 -> 2 (weed), keep 0 and 1 as-is.
    """
    if mapping is None:
        out = np.array(label, copy=True)
        out[out > 1] = 2
        return out.astype(np.uint8)

    out = np.zeros_like(label, dtype=np.uint8)
    for k, v in mapping.items():
        k_i = int(k)
        v_i = int(v)
        out[label == k_i] = v_i
    return out


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_split_file(splits_dir, split_name):
    path = os.path.join(splits_dir, f"{split_name}.txt")
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def write_split_list(dst_root, split_name, ids):
    path = os.path.join(dst_root, f"{split_name}.txt")
    with open(path, 'w') as f:
        for _id in ids:
            f.write(f"{_id}\n")


def convert(args):
    splits_dir = args.splits_dir
    src = args.src
    dst = args.dst
    ensure_dir(dst)

    mapping = None
    if args.mapping:
        mapping = json.loads(args.mapping)

    splits = {'train': [], 'val': [], 'test': []}
    # If user provided split txts, read them; otherwise use the three standard files under src/splits
    for s in splits.keys():
        ids = []
        if splits_dir and os.path.isdir(splits_dir):
            ids = load_split_file(splits_dir, s)
        elif os.path.isdir(os.path.join(src, 'splits')):
            ids = load_split_file(os.path.join(src, 'splits'), s)
        splits[s] = ids

    # If none present, try to discover from directory
    all_ids = set()
    for s in splits:
        all_ids.update(splits[s])
    if not all_ids:
        # try to find any semantics files
        for root, _, files in os.walk(src):
            for fn in files:
                if fn.endswith('.png') and not fn.endswith(('_R.png', '_G.png', '_B.png', '_NIR.png', '_RE.png')):
                    # assume semantic file; take filename without ext
                    name = os.path.splitext(fn)[0]
                    all_ids.add(name)
        splits['train'] = sorted(list(all_ids))

    # Create output directories
    for s in ['train', 'val', 'test']:
        ensure_dir(os.path.join(dst, 'images', s))
        ensure_dir(os.path.join(dst, 'labels', s))

    # Process each split
    for s in ['train', 'val', 'test']:
        ids = splits.get(s, [])
        print(f"Converting {len(ids)} items for split '{s}'")
        for image_id in ids:
            try:
                rgb = build_rgb(src, image_id, in_bands=args.in_bands)
            except FileNotFoundError as e:
                print(f"Skipping {image_id} (missing band): {e}")
                continue
            except Exception as e:
                print(f"Error reading bands for {image_id}: {e}")
                continue

            try:
                label = read_label(src, image_id)
            except FileNotFoundError as e:
                print(f"Skipping {image_id} (missing label): {e}")
                continue

            new_label = remap_label(label, mapping=mapping)

            # Save image and label
            out_img_path = os.path.join(dst, 'images', s, image_id + '.png')
            out_lbl_path = os.path.join(dst, 'labels', s, image_id + '.png')

            Image.fromarray(to_uint8(rgb)).save(out_img_path)
            # Save label as single-channel PNG
            Image.fromarray(new_label).save(out_lbl_path)

        # Write split list file
        write_split_list(dst, s, ids)

    print('Conversion finished.')


def parse_args():
    p = argparse.ArgumentParser(description='Convert WeedsGalore-like dataset to RGB+labels with splits')
    p.add_argument('--src', required=True, help='Root of weeds_galore dataset')
    p.add_argument('--dst', required=True, help='Output root for converted dataset')
    p.add_argument('--splits-dir', default=None, help='Directory containing train.txt/val.txt/test.txt (optional)')
    p.add_argument('--in-bands', type=int, default=5, help='Number of input bands (3 or 5). Only R,G,B used for output RGB')
    p.add_argument('--mapping', default=None, help='JSON string describing label mapping e.g. "{\"0\":0, \"1\":1, \"2\":2}"')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(args)
