#!/usr/bin/env python3
"""
Minimalistic Segmentation Inference Script

A lightweight, modular inference script for segmentation models that minimizes
dependencies on MMSegmentation framework while maintaining flexibility to load
different model architectures (Mask2Former, etc.) using pretrained weights.

Usage:
    python minimal_inference.py --config config.py --checkpoint model.pth --input image.jpg
    python minimal_inference.py --model-type mask2former --checkpoint model.pth --input folder/
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import logging

# Minimal imports from mmengine/mmseg for essential functionality
try:
    from mmengine import Config
    from mmengine.runner import load_checkpoint
    HAS_MMENGINE = True
except ImportError:
    print("Warning: MMEngine not available. Some features may be limited.")
    HAS_MMENGINE = False

try:
    from mmseg.utils import register_all_modules
    from mmseg.models import build_segmentor
    HAS_MMSEG = True
except ImportError:
    print("Warning: MMSeg not available. Using fallback model loading.")
    HAS_MMSEG = False


class MinimalEncoderDecoder(nn.Module):
    """
    Minimalistic Encoder-Decoder architecture inspired by MMSeg's EncoderDecoder
    but with reduced dependencies and simplified interface.
    """
    
    def __init__(self, backbone=None, decode_head=None, neck=None, 
                 num_classes=3, align_corners=False):
        super().__init__()
        
        self.backbone = backbone
        self.decode_head = decode_head
        self.neck = neck
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        
        # Properties for compatibility
        self.with_neck = neck is not None
        self.with_decode_head = decode_head is not None
        
    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def encode_decode(self, inputs: torch.Tensor, batch_img_metas: List[dict]) -> torch.Tensor:
        """Encode images with backbone and decode into segmentation map."""
        x = self.extract_feat(inputs)
        
        # Handle different decode head types
        if hasattr(self.decode_head, 'predict_torchscript'):
            # Use TorchScript-compatible method if available
            height, width = batch_img_metas[0]['img_shape'][:2]
            seg_logits = self.decode_head.predict_torchscript(x, height, width)
        elif hasattr(self.decode_head, 'predict'):
            # Use standard predict method
            seg_logits = self.decode_head.predict(x, batch_img_metas, None)
        else:
            # Fallback to forward method
            seg_logits = self.decode_head(x)
            
        return seg_logits
    
    def forward(self, inputs: torch.Tensor, img_metas: Optional[List[dict]] = None) -> torch.Tensor:
        """Forward pass for inference."""
        if img_metas is None:
            # Create default metadata
            batch_size, _, height, width = inputs.shape
            img_metas = [{
                'img_shape': (height, width),
                'ori_shape': (height, width),
                'pad_shape': (height, width),
                'padding_size': [0, 0, 0, 0]
            }] * batch_size
            
        return self.encode_decode(inputs, img_metas)
    
    def inference(self, inputs: torch.Tensor, img_metas: List[dict]) -> torch.Tensor:
        """Main inference method."""
        return self.encode_decode(inputs, img_metas)


class ModelLoader:
    """Handles loading different model architectures with minimal dependencies."""
    
    @staticmethod
    def load_from_config(config_path: str, checkpoint_path: str, device: str = 'cuda') -> nn.Module:
        """Load model using MMSeg config and checkpoint."""
        if not HAS_MMSEG or not HAS_MMENGINE:
            raise ImportError("MMSeg and MMEngine required for config-based loading")
            
        register_all_modules()
        
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Build model
        model = build_segmentor(cfg.model)
        
        # Load checkpoint
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        
        # Convert SyncBatchNorm to BatchNorm2d for inference
        model = ModelLoader._convert_sync_batchnorm(model)
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        return model
    
    @staticmethod
    def load_torchscript(model_path: str, device: str = 'cuda') -> nn.Module:
        """Load TorchScript model."""
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    
    @staticmethod
    def create_mask2former_minimal(num_classes: int = 3, checkpoint_path: Optional[str] = None,
                                 device: str = 'cuda') -> MinimalEncoderDecoder:
        """Create a minimal Mask2Former model without full MMSeg dependency."""
        # This is a simplified version - you can expand this based on your needs
        
        # For now, this requires MMSeg for the actual components
        if not HAS_MMSEG:
            raise ImportError("MMSeg required for Mask2Former components")
            
        # You would implement simplified versions of backbone and head here
        # This is a placeholder that shows the structure
        raise NotImplementedError("Minimal Mask2Former implementation pending")
    
    @staticmethod
    def _convert_sync_batchnorm(module: nn.Module) -> nn.Module:
        """Convert SyncBatchNorm to BatchNorm2d for inference."""
        module_output = module
        if isinstance(module, torch.nn.SyncBatchNorm):
            module_output = torch.nn.BatchNorm2d(
                module.num_features, module.eps, module.momentum, 
                module.affine, module.track_running_stats)
            if module.affine:
                module_output.weight.data = module.weight.data.clone().detach()
                module_output.bias.data = module.bias.data.clone().detach()
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, ModelLoader._convert_sync_batchnorm(child))
        del module
        return module_output


class ImageProcessor:
    """Handles image preprocessing and postprocessing."""
    
    def __init__(self, mean: List[float] = [123.675, 116.28, 103.53],
                 std: List[float] = [58.395, 57.12, 57.375],
                 bgr_to_rgb: bool = True,
                 size_divisor: int = 32):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.bgr_to_rgb = bgr_to_rgb
        self.size_divisor = size_divisor
    
    def preprocess(self, img_path: str, target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, dict]:
        """Preprocess image for inference."""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
            
        if self.bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original shape
        ori_h, ori_w = img.shape[:2]
        
        # Resize if target size is specified
        if target_size is not None:
            img = cv2.resize(img, target_size)
        
        # Get current dimensions
        h, w = img.shape[:2]
        
        # Pad to make dimensions divisible by size_divisor
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        img_meta = dict()
        # Create metadata
        img_meta.update(
            img_shape=(h + pad_h, w + pad_w),
            ori_shape=(ori_h, ori_w),
            pad_shape=(h + pad_h, w + pad_w),
            padding_size=[0, pad_h, 0, pad_w],
            scale_factor=1.0,
            flip=False,
            filename=os.path.basename(img_path)
        )
        
        return img_tensor, img_meta
    
    def postprocess(self, seg_logits: torch.Tensor, img_meta: dict, 
                   original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Postprocess segmentation output."""
        # Get segmentation map
        seg_maps = []
        for seg_map_item in seg_logits:
            seg_map = seg_map_item.pred_sem_seg.data[0, :, :].cpu().numpy()
            seg_maps.append(seg_map)
        return seg_maps


class Visualizer:
    """Handles visualization of segmentation results."""
    
    def __init__(self, class_names: List[str] = ['Background', 'Crop', 'Weed'],
                 palette: Optional[List[List[int]]] = None):
        self.class_names = class_names
        if palette is None:
            self.palette = np.array([
                [0, 0, 0],        # Background: black
                [0, 255, 0],      # Crop: green  
                [255, 0, 0],      # Weed: red
            ], dtype=np.uint8)
        else:
            self.palette = np.array(palette, dtype=np.uint8)
    
    def visualize(self, img_path: str, seg_map: np.ndarray, save_path: Optional[str] = None,
                 show: bool = False, alpha: float = 0.5) -> np.ndarray:
        """Visualize segmentation results."""
        # Load original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize segmentation map to match image size if needed
        if seg_map.shape[:2] != img_rgb.shape[:2]:
            seg_map = cv2.resize(seg_map, (img_rgb.shape[1], img_rgb.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Create colored segmentation map
        color_seg = self.palette[seg_map]
        
        # Create overlay
        overlay = cv2.addWeighted(img_rgb, 1-alpha, color_seg, alpha, 0)
        
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(img_rgb)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Segmentation Map")
            plt.imshow(color_seg)
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(overlay)
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        if save_path:
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, overlay_bgr)
            print(f"Saved visualization to: {save_path}")
        
        return overlay


class SegmentationInferencer:
    """Main inference class that orchestrates the entire pipeline."""
    
    def __init__(self, model: nn.Module, processor: ImageProcessor, 
                 visualizer: Optional[Visualizer] = None, device: str = 'cuda'):
        self.model = model
        self.processor = processor
        self.visualizer = visualizer or Visualizer()
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def infer_single(self, img_path: str, target_size: Optional[Tuple[int, int]] = None,
                    save_path: Optional[str] = None, show: bool = False) -> Tuple[np.ndarray, float]:
        """Infer on a single image."""
        import time
        
        # Preprocess
        img_tensor, img_meta = self.processor.preprocess(img_path, target_size)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            seg_logits = self.model(img_tensor, mode='predict')
        inference_time = time.time() - start_time
        
        # Postprocess
        seg_map = self.processor.postprocess(seg_logits, img_meta)
        
        # Visualize if requested
        if save_path or show:
            self.visualizer.visualize(img_path, seg_map[0], save_path, show)
        
        return seg_map, inference_time
    
    def infer_folder(self, folder_path: str, target_size: Optional[Tuple[int, int]] = None,
                    output_dir: Optional[str] = None, extensions: List[str] = ['jpg', 'png', 'jpeg']):
        """Infer on all images in a folder."""
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext.upper()}')))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        total_time = 0
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            save_path = None
            if output_dir:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(output_dir, f"{base_name}_seg.png")
            
            try:
                seg_map, inference_time = self.infer_single(img_path, target_size, save_path)
                total_time += inference_time
                print(f"  Inference time: {inference_time:.3f}s ({1/inference_time:.1f} FPS)")
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
        
        if len(image_files) > 0:
            avg_time = total_time / len(image_files)
            print(f"\nAverage inference time: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Minimal Segmentation Inference")
    
    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--config', type=str, help='Path to config file')
    model_group.add_argument('--torchscript', type=str, help='Path to TorchScript model')
    model_group.add_argument('--model-type', type=str, choices=['mask2former'], 
                           help='Model type for minimal loading')
    
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image or folder path')
    parser.add_argument('--output', type=str, help='Output directory for visualizations')
    parser.add_argument('--target-size', type=int, nargs=2, 
                       help='Target size for resizing (height width)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    if args.config:
        if not args.checkpoint:
            raise ValueError("Checkpoint required when using config")
        model = ModelLoader.load_from_config(args.config, args.checkpoint, args.device)
    elif args.torchscript:
        model = ModelLoader.load_torchscript(args.torchscript, args.device)
    elif args.model_type:
        if not args.checkpoint:
            raise ValueError("Checkpoint required when using model-type")
        if args.model_type == 'mask2former':
            model = ModelLoader.create_mask2former_minimal(args.num_classes, args.checkpoint, args.device)
    
    # Create processor and visualizer
    processor = ImageProcessor()
    visualizer = Visualizer()
    
    # Create inferencer
    inferencer = SegmentationInferencer(model, processor, visualizer, args.device)
    
    # Run inference
    target_size = tuple(args.target_size) if args.target_size else None
    
    if os.path.isfile(args.input):
        print(f"Processing single image: {args.input}")
        save_path = None
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            save_path = os.path.join(args.output, f"{base_name}_seg.png")
        
        seg_map, inference_time = inferencer.infer_single(args.input, target_size, save_path, args.show)
        print(f"Inference time: {inference_time:.3f}s ({1/inference_time:.1f} FPS)")
        
    elif os.path.isdir(args.input):
        print(f"Processing folder: {args.input}")
        inferencer.infer_folder(args.input, target_size, args.output)
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    print("Done!")


if __name__ == '__main__':
    main()
