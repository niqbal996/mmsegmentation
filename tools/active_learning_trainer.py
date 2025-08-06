#!/usr/bin/env python3
"""
Active Learning Trainer with Score Generator

A compact active learning trainer that generates uncertainty scores for 
Phenobench dataset using Mask2Former models. Outputs sorted image rankings
based on floating region scores for active learning sample selection.

Usage:
    python active_learning_trainer.py --config config.py --checkpoint model.pth --data-root /path/to/phenobench
"""

import os
import cv2
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# Custom palette: 0=black, 1=green, 2=red
palette = np.array([
    [0, 0, 0],      # 0: black
    [0, 255, 0],    # 1: green
    [255, 0, 0]     # 2: red
], dtype=np.uint8)

# MMSeg imports
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmseg.utils import register_all_modules
from mmseg.models import build_segmentor
from mmseg.datasets.phenobench_dataset import PhenoBench_processed
from mmseg.structures import SegDataSample
from torch.utils.data import DataLoader

# Local imports
from active_learning import FloatingRegionScore, RegionSelection


class ActiveLearningScoreGenerator:
    def __init__(self, 
                 model: torch.nn.Module,
                 config: Config,
                 device: str = 'cuda',
                 num_classes: int = 3,
                 ):
        """
        Initialize score generator.
        
        Args:
            model: Pretrained segmentation model
            device: Device to run inference on
            num_classes: Number of segmentation classes
            region_size: Size of floating regions for score calculation
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.config = config
        self.radius_K = config['RADIUS_K']
        self.ratio = config['RATIO']
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Initialize floating region score calculator
        self.floating_score = FloatingRegionScore(
            in_channels=num_classes, 
            size=2*self.radius_K + 1
        ).to(device)
    
    def get_all_active_regions_mask(self, 
                                    scores: torch.Tensor,
                                    groundtruth_mask: np.ndarray,
                                    active_mask: np.ndarray,
                                    active_regions: int,
                                    active: torch.Tensor,
                                    selected: torch.Tensor):
        """
        Return a binary mask of all active regions selected by the region selection logic.
        This function implements the same logic as the original RegionSelection but for offline processing.
        
        Args:
            scores: torch.Tensor, informativeness score map (H, W)
            groundtruth_mask: np.ndarray, ground truth segmentation mask
            active_mask: np.ndarray, current active mask to update
            active_regions: int, number of regions to select
            active: torch.Tensor, indicator for masked out regions
            selected: torch.Tensor, indicator for selected regions
            
        Returns:
            tuple: (updated_active_mask, active_indicator, selected_indicator) as numpy arrays
        """
        # Clone scores to avoid modifying original
        scores = scores.clone().detach()
        groundtruth_mask = torch.tensor(groundtruth_mask, dtype=torch.bool).to(self.device)
        
        # Ensure tensors are properly shaped and on correct device
        active = active.squeeze(0).to(self.device)
        selected = selected.squeeze(0).to(self.device)
        active_mask = torch.tensor(active_mask.squeeze(0), dtype=torch.bool).to(self.device)
        
        H, W = scores.shape
        
        # Set scores of already active regions to -inf (same as original RegionSelection)
        scores[active] = -float('inf')
        
        # Iteratively select regions with highest scores
        for pixel in range(active_regions):
            # Find location with maximum score
            values, indices_h = torch.max(scores, dim=0)
            _, indices_w = torch.max(values, dim=0)
            w = indices_w.item()
            h = indices_h[w].item()
            
            # Calculate active region bounds (region that gets labeled)
            active_start_w = w - self.active_radius if w - self.active_radius >= 0 else 0
            active_start_h = h - self.active_radius if h - self.active_radius >= 0 else 0
            active_end_w = min(w + self.active_radius + 1, W)
            active_end_h = min(h + self.active_radius + 1, H)
            
            # Calculate mask region bounds (larger region that gets masked out)
            mask_start_w = w - self.mask_radius if w - self.mask_radius >= 0 else 0
            mask_start_h = h - self.mask_radius if h - self.mask_radius >= 0 else 0
            mask_end_w = min(w + self.mask_radius + 1, W)
            mask_end_h = min(h + self.mask_radius + 1, H)
            
            # Mask out the larger region to prevent overlapping selections
            scores[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
            active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
            selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
            
            # Update active mask with ground truth labels for the selected region
            active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        groundtruth_mask[active_start_h:active_end_h, active_start_w:active_end_w]
        
        return active_mask.cpu().numpy(), active.cpu().numpy(), selected.cpu().numpy()

    def get_regions_above_threshold(self, 
                                   scores: torch.Tensor,
                                   groundtruth_mask: np.ndarray,
                                   active_mask: np.ndarray,
                                   threshold: float,
                                   active: torch.Tensor,
                                   selected: torch.Tensor,
                                   max_regions: int = None):
        """
        Select all regions above a certain uncertainty threshold.
        
        Args:
            scores: torch.Tensor, informativeness score map (H, W)
            groundtruth_mask: np.ndarray, ground truth segmentation mask
            active_mask: np.ndarray, current active mask to update
            threshold: float, minimum score threshold for region selection
            active: torch.Tensor, indicator for masked out regions
            selected: torch.Tensor, indicator for selected regions
            max_regions: int, optional maximum number of regions to select
            
        Returns:
            tuple: (updated_active_mask, active_indicator, selected_indicator) as numpy arrays
        """
        # Clone scores to avoid modifying original
        scores = scores.clone().detach()
        groundtruth_mask = torch.tensor(groundtruth_mask, dtype=torch.bool).to(self.device)
        
        # Ensure tensors are properly shaped and on correct device
        active = active.squeeze(0).to(self.device)
        selected = selected.squeeze(0).to(self.device)
        active_mask = torch.tensor(active_mask.squeeze(0), dtype=torch.bool).to(self.device)
        
        H, W = scores.shape
        
        # Set scores of already active regions to -inf
        scores[active] = -float('inf')
        
        regions_selected = 0
        
        # Continue selecting regions while above threshold
        while True:
            # Find location with maximum score
            max_score = torch.max(scores)
            
            # Break if below threshold or max regions reached
            if max_score < threshold or max_score == -float('inf'):
                break
            if max_regions is not None and regions_selected >= max_regions:
                break
                
            values, indices_h = torch.max(scores, dim=0)
            _, indices_w = torch.max(values, dim=0)
            w = indices_w.item()
            h = indices_h[w].item()
            
            # Calculate region bounds
            active_start_w = w - self.active_radius if w - self.active_radius >= 0 else 0
            active_start_h = h - self.active_radius if h - self.active_radius >= 0 else 0
            active_end_w = min(w + self.active_radius + 1, W)
            active_end_h = min(h + self.active_radius + 1, H)
            
            mask_start_w = w - self.mask_radius if w - self.mask_radius >= 0 else 0
            mask_start_h = h - self.mask_radius if h - self.mask_radius >= 0 else 0
            mask_end_w = min(w + self.mask_radius + 1, W)
            mask_end_h = min(h + self.mask_radius + 1, H)
            
            # Mask out the region
            scores[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
            active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
            selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
            
            # Update active mask with ground truth
            active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        groundtruth_mask[active_start_h:active_end_h, active_start_w:active_end_w]
            
            regions_selected += 1
        
        return active_mask.cpu().numpy(), active.cpu().numpy(), selected.cpu().numpy()

    def visualize_results(self, mask_dict: Dict[str, np.ndarray], save_path=None):
        """
        Visualize all tensors in mask_dict in a 2x3 grid (enlarged for detail).
        Args:
            mask_dict: dict containing all tensors to visualize (scores, predicted_mask, region_impurity, prediction_uncertainty, gt_mask, active_mask)
            save_path: Optional path to save visualization
        """
        # Unpack tensors from dict
        scores = mask_dict.get('scores')
        predicted_mask = mask_dict.get('predicted_mask')
        region_impurity = mask_dict.get('region_impurity')
        prediction_uncertainty = mask_dict.get('prediction_uncertainty')
        gt_mask = mask_dict.get('gt_mask')
        active_mask = mask_dict.get('active_mask')

        # Prepare scores as numpy
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy().squeeze()
        else:
            scores_np = np.array(scores).squeeze()

        fig, axs = plt.subplots(2, 3, figsize=(21, 14))
        # Top-left: Ground Truth
        if gt_mask is not None:
            gt_mask_rgb = palette[gt_mask.clip(0, 2)]
            axs[0, 0].imshow(gt_mask_rgb)
            axs[0, 0].set_title('Ground Truth', fontsize=18)
        else:
            axs[0, 0].axis('off')
        axs[0, 0].axis('off')

        # Top-middle: Prediction
        pred_mask_rgb = palette[predicted_mask.clip(0, 2)]
        axs[0, 1].imshow(pred_mask_rgb)
        axs[0, 1].set_title('Prediction', fontsize=18)
        axs[0, 1].axis('off')

        # Top-right: Scores heatmap
        im_score = axs[0, 2].imshow(scores_np, cmap='viridis', norm=Normalize(vmin=scores_np.min(), vmax=scores_np.max()))
        axs[0, 2].set_title(f'Scores\nmin={scores_np.min():.3e}, max={scores_np.max():.3e}, mean={scores_np.mean():.3e}', fontsize=16)
        axs[0, 2].axis('off')
        fig.colorbar(im_score, ax=axs[0, 2], fraction=0.046, pad=0.04)

        # Bottom-left: Region Impurity
        im2 = axs[1, 0].imshow(region_impurity, cmap='hot', norm=Normalize(vmin=region_impurity.min(), vmax=region_impurity.max()))
        axs[1, 0].set_title('Region Impurity (P)', fontsize=18)
        axs[1, 0].axis('off')
        fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

        # Bottom-middle: Prediction Uncertainty
        im3 = axs[1, 1].imshow(prediction_uncertainty, cmap='hot', norm=Normalize(vmin=prediction_uncertainty.min(), vmax=prediction_uncertainty.max()))
        axs[1, 1].set_title('Prediction Uncertainty (U)', fontsize=18)
        axs[1, 1].axis('off')
        fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

        # Bottom-right: All active regions binary mask
        if active_mask is not None:
            # Convert boolean mask to uint8 for visualization
            mask_vis = active_mask.astype(np.uint8)
            axs[1, 2].imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
            axs[1, 2].set_title('All selected regions', fontsize=16)
        else:
            axs[1, 2].axis('off')

        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f'Saved visualization to {save_path}')

        # Use keypress event to advance (space/right arrow = next, q = quit)
        self._skip_flag = False
        def on_key(event):
            if event.key in [' ', 'right']:
                self._skip_flag = True
                plt.close(fig)
            elif event.key == 'q':
                self._skip_flag = 'quit'
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        if self._skip_flag == 'quit':
            return False
        return True

    def get_active_region_mask(self, scores, region_size=11):
        """
        Return a binary mask (black/white) of the active region (max score region).
        Args:
            scores: np.ndarray, informativeness score map (H, W)
            region_size: int, size of the region to highlight (default: 11)
        Returns:
            np.ndarray: binary mask (H, W), 1 for active region, 0 elsewhere
        """
        import numpy as np
        h, w = np.unravel_index(np.argmax(scores), scores.shape)
        mask = np.zeros_like(scores, dtype=np.uint8)
        half = region_size // 2
        y1, y2 = max(0, h - half), min(scores.shape[0], h + half + 1)
        x1, x2 = max(0, w - half), min(scores.shape[1], w + half + 1)
        mask[y1:y2, x1:x2] = 1
        return mask

    def visualize_active_region_on_prediction(self, scores, pred_mask_rgb, region_size=11):
        """
        Overlay the highest-score active region on the prediction mask.
        Args:
            scores: np.ndarray, informativeness score map (H, W)
            pred_mask_rgb: np.ndarray, RGB prediction mask (H, W, 3)
            region_size: int, size of the region to highlight (default: 11)
        Returns:
            np.ndarray: RGB image with active region highlighted
        """
        import cv2
        import numpy as np
        # Find max score location
        h, w = np.unravel_index(np.argmax(scores), scores.shape)
        overlay = pred_mask_rgb.copy()
        half = region_size // 2
        y1, y2 = max(0, h - half), min(scores.shape[0], h + half + 1)
        x1, x2 = max(0, w - half), min(scores.shape[1], w + half + 1)
        # Draw a red rectangle (or transparent overlay)
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 255), thickness=2)
        alpha = 0.4
        region = overlay[y1:y2, x1:x2]
        highlight = np.full(region.shape, (255, 0, 255), dtype=np.uint8)
        cv2.addWeighted(highlight, alpha, region, 1 - alpha, 0, region)
        overlay[y1:y2, x1:x2] = region
        return overlay
    """
    Score generator for active learning using floating region scores.
    """
    def generate_scores(self, 
                       dataset: torch.utils.data.Dataset,
                       batch_size: int = 1,
                       score_type: str = 'entropy') -> List[Dict]:
        """
        Generate uncertainty scores for all images in dataset.
        
        Args:
            dataset: Phenobench dataset
            batch_size: Batch size for inference
            score_type: Type of score ('entropy', 'purity', or 'mixed')
            
        Returns:
            List of dictionaries with image paths and scores
        """
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=custom_collate)
        
        self.region_size = 2*self.config.get('RADIUS_K', 1)+1
        self.per_region_pixels = self.region_size ** 2
        self.active_radius = self.config.get('RADIUS_K', 1)
        self.mask_radius = self.active_radius * 2
        self.active_ratio = self.config.get('RATIO', 1) / 10000  # TODO What are these iterator values in offline AL mode.

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating scores"):
                # Extract batch data
                imgs = batch['inputs'].to(self.device)
                origin_mask, origin_label, active_indicator, selected_indicator = \
                    batch['origin_mask'], batch['origin_label'], \
                    batch['active'], batch['selected']
                gt_sem_seg = batch['gt_sem_seg'][0].data    # This is same as origin_label
                img_metas = batch['metainfo']
                
                # Forward pass
                features = self.model.extract_feat(imgs)
                # segmentation_logits are already interpolated to the source image size
                segmentation_logits = self.model.decode_head.predict(features, 
                                                                   batch_img_metas=batch['metainfo'],
                                                                   test_cfg=self.model.decode_head.test_cfg)
                # Convert batch metainfo into a SegDataSample object
                seg_data_sample = SegDataSample()
                seg_data_sample.set_metainfo(batch['metainfo'][0])
                predicted_mask = self.model.postprocess_result(segmentation_logits, [seg_data_sample])
                num_pixel_curr = imgs.shape[2] * imgs.shape[3]
                active_regions = math.ceil(num_pixel_curr * self.active_ratio / self.per_region_pixels)
                # Process each item in batch
                for i, (logits, meta) in enumerate(zip(segmentation_logits, img_metas)):
                    # Calculate floating region scores for this sample
                    scores, region_impurity_P, prediction_uncertainty_U = self.floating_score(logits.unsqueeze(0))
                    predicted_mask = predicted_mask[i].pred_sem_seg.data.cpu().numpy()[0]
                    region_impurity_P = region_impurity_P.cpu().numpy().squeeze()
                    prediction_uncertainty_U = prediction_uncertainty_U.cpu().numpy().squeeze()
                    gt_mask_np = gt_sem_seg.cpu().numpy()[0] if gt_sem_seg is not None else None
                    
                    # Set scores of already active regions to -inf
                    scores[active_indicator[i]] = -float('inf')
                    
                    # Choose region selection method
                    if hasattr(self, 'selection_mode') and self.selection_mode == 'threshold':
                        if not hasattr(self, 'uncertainty_threshold') or self.uncertainty_threshold is None:
                            raise ValueError("uncertainty_threshold must be set when using threshold selection mode")
                        
                        active_mask, active, selected = self.get_regions_above_threshold(
                            scores=scores,
                            groundtruth_mask=gt_mask_np,
                            active_mask=origin_mask,
                            threshold=self.uncertainty_threshold,
                            active=active_indicator,
                            selected=selected_indicator,
                            max_regions=getattr(self, 'max_regions_per_image', None)
                        )
                    else:
                        # Default: ratio-based selection (original method)
                        active_mask, active, selected = self.get_all_active_regions_mask(
                            scores=scores,
                            groundtruth_mask=gt_mask_np,
                            active_mask=origin_mask,
                            active_regions=active_regions,
                            active=active_indicator,
                            selected=selected_indicator
                        )
                    curr_visualization = {
                        # 'img_path': meta['img_path'],
                        'scores': scores,
                        'predicted_mask': predicted_mask,
                        'region_impurity': region_impurity_P,
                        'prediction_uncertainty': prediction_uncertainty_U,
                        'gt_mask': gt_mask_np,
                        'active_mask': active  # Convert to binary mask
                    }
                    cont = self.visualize_results(curr_visualization)
                    indicator = {
                        'active': active,
                        'selected': selected,
                    }
                    # torch.save(indicator, os.path.join(batch['path_to_indicator'], meta['filename'][:-4] + '.pt'))
                    if cont is False:
                        break
    
    def save_ranked_results(self, 
                           results: List[Dict],
                           output_path: str,
                           sort_ascending: bool = False,
                           format: str = 'json') -> str:
        """
        Save ranked results to file.
        
        Args:
            results: List of score results
            output_path: Output file path
            sort_ascending: If True, sort from lowest to highest score
            format: Output format ('json' or 'csv')
            
        Returns:
            Path to saved file
        """
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=not sort_ascending)
        
        # Add ranking
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1
        
        # Save to file
        if format.lower() == 'json':
            output_file = f"{output_path}.json"
            with open(output_file, 'w') as f:
                json.dump(sorted_results, f, indent=2)
        elif format.lower() == 'csv':
            output_file = f"{output_path}.csv"
            with open(output_file, 'w', newline='') as f:
                if sorted_results:
                    writer = csv.DictWriter(f, fieldnames=sorted_results[0].keys())
                    writer.writeheader()
                    writer.writerows(sorted_results)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {len(sorted_results)} ranked results to {output_file}")
        return output_file
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        if hasattr(self, 'feature_hook'):
            self.feature_hook.remove_hooks()


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Load model from config and checkpoint."""
    register_all_modules()
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = build_segmentor(cfg.model)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Move to device
    model.to(device)
    model.eval()
    
    return model


def custom_collate(batch):
    # batch is a list of dicts
    out = {}
    for key in batch[0]:
        if key == 'data_samples':
            out[key] = [b[key] for b in batch]  # keep as list
        else:
            out[key] = torch.stack([b[key] for b in batch]) if isinstance(batch[0][key], torch.Tensor) else [b[key] for b in batch]
    return out


def main():
    parser = argparse.ArgumentParser(description="Active Learning Score Generator")
    parser.add_argument('--config', type=str, required=True, help='Model config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Phenobench dataset root')
    parser.add_argument('--output', type=str, default='active_learning_scores', 
                       help='Output file prefix')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--score-type', type=str, default='entropy', 
                       choices=['entropy', 'purity', 'mixed'], help='Score type')
    parser.add_argument('--region-size', type=int, default=5, help='Floating region size (region_size)')
    parser.add_argument('--mask-radius', type=int, default=None, help='Mask radius for region suppression (mask_radius, default: 2*region_size-1)')
    parser.add_argument('--active-ratio', type=float, default=0.01, help='Active ratio (fraction of image to select as regions, e.g. 0.01)')
    parser.add_argument('--selection-mode', type=str, default='ratio', choices=['ratio', 'threshold'],
                       help='Region selection mode: ratio (fixed number based on image size) or threshold (all regions above threshold)')
    parser.add_argument('--uncertainty-threshold', type=float, default=None,
                       help='Uncertainty threshold for threshold-based selection (required if selection-mode=threshold)')
    parser.add_argument('--max-regions-per-image', type=int, default=None,
                       help='Maximum regions per image (optional limit for threshold-based selection)')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv'],
                       help='Output format')
    parser.add_argument('--sort-ascending', action='store_true',
                       help='Sort from lowest to highest score (default: highest to lowest)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.selection_mode == 'threshold' and args.uncertainty_threshold is None:
        parser.error("--uncertainty-threshold is required when --selection-mode=threshold")
    
    # Load model
    print("Loading model...")
    model = load_model(args.config, args.checkpoint, args.device)
    
    # Create dataset config
    data_cfg = {
        'type': 'PhenobenchDataset',
        'data_root': args.data_root,
        'data_prefix': {
            'img_path': 'images',
            'seg_map_path': 'labels'
        },
        'pipeline': [
            {'type': 'LoadImageFromFile'},
            {'type': 'LoadAnnotations'},
            {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': True},
            {'type': 'PackSegInputs'}
        ],
        'RATIO': 0.05,
        'RADIUS_K': 1
    }
    
    data_cfg['data_prefix']['seg_map_path'] = 'semantics'
    # Build dataset
    print("Building dataset...")
    dataset = PhenoBench_processed(data_root=os.path.join(data_cfg['data_root'], 'train'),
                                data_prefix=data_cfg['data_prefix'])
    print(f"Dataset size: {len(dataset)}")
    
    # Create score generator
    score_generator = ActiveLearningScoreGenerator(
        model=model,
        device=args.device,
        num_classes=3,  # Background, Crop, Weed
        config=data_cfg
    )
    
    # Set selection parameters
    score_generator.selection_mode = args.selection_mode
    if args.selection_mode == 'threshold':
        score_generator.uncertainty_threshold = args.uncertainty_threshold
        score_generator.max_regions_per_image = args.max_regions_per_image
    
    # Override active ratio if provided
    if args.active_ratio != 0.01:  # If user provided a different value
        score_generator.active_ratio = args.active_ratio / 100
    
    # Print selection configuration
    print(f"\nRegion Selection Configuration:")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Region size (radius_K): {data_cfg['RADIUS_K']}")
    print(f"Active radius: {score_generator.active_radius}")
    print(f"Mask radius: {score_generator.mask_radius}")
    
    if args.selection_mode == 'ratio':
        print(f"Active ratio: {score_generator.active_ratio:.4f}")
        print(f"Regions per image (512x512): ~{math.ceil(512*512 * score_generator.active_ratio / score_generator.per_region_pixels)}")
    else:
        print(f"Uncertainty threshold: {args.uncertainty_threshold}")
        if args.max_regions_per_image:
            print(f"Max regions per image: {args.max_regions_per_image}")
        else:
            print("Max regions per image: unlimited")
    print()
    
    # Generate scores
    print("Generating uncertainty scores...")
    results = score_generator.generate_scores(
        dataset=dataset,
        batch_size=args.batch_size,
        score_type=args.score_type
    )
    
    # Save results
    print("Saving ranked results...")
    output_file = score_generator.save_ranked_results(
        results=results,
        output_path=args.output,
        sort_ascending=args.sort_ascending,
        format=args.format
    )
    
    # Print summary statistics
    scores = [r['score'] for r in results]
    print(f"\nSummary Statistics:")
    print(f"Total images: {len(results)}")
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Std score: {np.std(scores):.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
