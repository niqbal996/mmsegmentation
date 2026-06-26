#!/usr/bin/env python3
"""
Compute Fréchet Inception Distance (FID) across three datasets:
  - real1  : Phenobench (primary real dataset, split into acquisition-day subsets)
  - real2  : CropAndWeed (second real dataset — real-to-real baseline)
  - syn    : SugarBeet Synthetic 2026

FID always compares two image distributions using 2048-dim pool3 features from
an InceptionV3 pretrained on ImageNet. Lower FID = more similar distributions.

Comparisons:
  real-to-synthetic  (domain-gap measurement)
    phenobench_all  vs synthetic
    phenobench_05-15 / 05-26 / 06-05  vs synthetic

  real-to-real  (baseline — how far two real datasets are from each other)
    phenobench_all  vs cropandweed
    phenobench_05-15 / 05-26 / 06-05  vs cropandweed

  optional --cross-day
    intra-phenobench day-subset pairs

Usage:
  python compute_fid.py
  python compute_fid.py --real1-dir /mnt/e/datasets/phenobench \
                        --real2-dir /ds/images/cropandweed/images \
                        --synthetic-dir /netscratch/naeem/sugarbeet_syn_v6 \
                        --splits train val \
                        --cross-day
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageListDataset(Dataset):
    """Loads images from an explicit list of file paths."""

    _transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # Scale [0,1] → [-1, 1] as expected by the original FID implementation
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self._transform(Image.open(self.paths[idx]).convert('RGB'))


# ---------------------------------------------------------------------------
# Inception feature extractor (pool3 → 2048-dim)
# ---------------------------------------------------------------------------

class InceptionPool3(nn.Module):
    """InceptionV3 truncated at the global average pool (2048-dim output)."""

    def __init__(self):
        super().__init__()
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        net.eval()
        # Build the feature backbone in order, stopping after Mixed_7c
        self.backbone = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.backbone(x).flatten(1)  # (B, 2048)


# ---------------------------------------------------------------------------
# Core FID utilities
# ---------------------------------------------------------------------------

def collect_paths(folder, exts=('*.png', '*.jpg', '*.jpeg')):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


@torch.no_grad()
def extract_features(model, paths, batch_size, device, label=''):
    if not paths:
        raise ValueError(f"No images found for '{label}'")
    loader = DataLoader(
        ImageListDataset(paths),
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.startswith('cuda'),
    )
    feats = []
    desc = f"  features [{label}] ({len(paths)} imgs)"
    for batch in tqdm(loader, desc=desc, ncols=90):
        feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)  # (N, 2048)


def gaussian_params(features):
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def fid_from_params(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    # Numerically stable matrix square root
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Square root of covariance product has large imaginary part")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid(feats_a, feats_b):
    return fid_from_params(*gaussian_params(feats_a), *gaussian_params(feats_b))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--real1-dir', default='/mnt/e/datasets/phenobench',
                   help='Root of Phenobench (real1); images at <root>/<split>/images/')
    p.add_argument('--real2-dir', default='/ds/images/cropandweed/images',
                   help='Flat images folder for CropAndWeed (real2)')
    p.add_argument('--synthetic-dir', default='/netscratch/naeem/sugarbeet_syn_v6',
                   help='Root of synthetic dataset; images at <root>/images/<split>/')
    p.add_argument('--splits', nargs='+', default=['train', 'val'],
                   choices=['train', 'val', 'test'],
                   help='Splits to include for real1 and synthetic (real2 has no splits)')
    p.add_argument('--day-prefixes', nargs='+', default=['05-15', '05-26', '06-05'])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cross-day', action='store_true',
                   help='Also compute FID between every pair of day subsets')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {args.device}")

    # ------------------------------------------------------------------
    # Collect image paths
    # ------------------------------------------------------------------
    # real1 — Phenobench (split-structured)
    real1_paths = []
    for split in args.splits:
        folder = os.path.join(args.real1_dir, split, 'images')
        found = collect_paths(folder)
        print(f"  phenobench / {split}: {len(found)} images  ({folder})")
        real1_paths.extend(found)

    # real2 — CropAndWeed (flat folder, no splits)
    real2_paths = collect_paths(args.real2_dir)
    print(f"  cropandweed:          {len(real2_paths)} images  ({args.real2_dir})")

    # synthetic
    syn_paths = []
    for split in args.splits:
        folder = os.path.join(args.synthetic_dir, 'images', split)
        found = collect_paths(folder)
        print(f"  synthetic / {split}:   {len(found)} images  ({folder})")
        syn_paths.extend(found)

    if not real1_paths:
        sys.exit("No Phenobench images found — check --real1-dir and --splits")
    if not real2_paths:
        sys.exit("No CropAndWeed images found — check --real2-dir")
    if not syn_paths:
        sys.exit("No synthetic images found — check --synthetic-dir and --splits")

    # Per-day subsets of Phenobench
    day_paths = {}
    for prefix in args.day_prefixes:
        subset = [p for p in real1_paths if os.path.basename(p).startswith(prefix)]
        day_paths[prefix] = subset
        print(f"  phenobench / {prefix}: {len(subset)} images")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading InceptionV3 (ImageNet weights)…")
    model = InceptionPool3().to(args.device)
    model.eval()

    # ------------------------------------------------------------------
    # Extract features (each set extracted once, reused across comparisons)
    # ------------------------------------------------------------------
    print("\nExtracting features…")
    syn_feats   = extract_features(model, syn_paths,   args.batch_size, args.device, 'synthetic')
    real1_feats = extract_features(model, real1_paths, args.batch_size, args.device, 'phenobench_all')
    real2_feats = extract_features(model, real2_paths, args.batch_size, args.device, 'cropandweed')

    day_feats = {}
    for prefix, paths in day_paths.items():
        if paths:
            day_feats[prefix] = extract_features(model, paths, args.batch_size,
                                                  args.device, f'phenobench_{prefix}')

    results = {}

    # ------------------------------------------------------------------
    # Real-to-synthetic (domain gap)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FID  real → synthetic  (domain gap)")
    print("=" * 60)

    results['phenobench_all__vs__synthetic'] = fid(real1_feats, syn_feats)
    print(f"  {'phenobench_all':<22}  {results['phenobench_all__vs__synthetic']:>8.2f}")

    for prefix, feats in day_feats.items():
        key = f'phenobench_{prefix}__vs__synthetic'
        results[key] = fid(feats, syn_feats)
        print(f"  {f'phenobench_{prefix}':<22}  {results[key]:>8.2f}")

    # ------------------------------------------------------------------
    # Real-to-real (baseline — how different two real datasets look)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FID  real → real  (baseline)")
    print("=" * 60)

    results['phenobench_all__vs__cropandweed'] = fid(real1_feats, real2_feats)
    print(f"  {'phenobench_all':<22}  {results['phenobench_all__vs__cropandweed']:>8.2f}")

    for prefix, feats in day_feats.items():
        key = f'phenobench_{prefix}__vs__cropandweed'
        results[key] = fid(feats, real2_feats)
        print(f"  {f'phenobench_{prefix}':<22}  {results[key]:>8.2f}")

    # ------------------------------------------------------------------
    # Optional: intra-Phenobench cross-day
    # ------------------------------------------------------------------
    if args.cross_day and len(day_feats) > 1:
        print("\n" + "=" * 60)
        print("FID  day subset → day subset  (intra-Phenobench variability)")
        print("=" * 60)
        prefixes = list(day_feats.keys())
        for i in range(len(prefixes)):
            for j in range(i + 1, len(prefixes)):
                pa, pb = prefixes[i], prefixes[j]
                key = f'phenobench_{pa}__vs__{pb}'
                results[key] = fid(day_feats[pa], day_feats[pb])
                print(f"  {pa}  vs  {pb:<10}  {results[key]:>8.2f}")

    print("\nNote: lower FID = more similar distributions.")
    print("FID estimates with < ~2 000 images can be noisy.")

    return results


if __name__ == '__main__':
    main()
