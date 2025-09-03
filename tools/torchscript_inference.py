#!/usr/bin/env python3
"""
TorchScript Wrapper for Minimal Inference

This script provides a wrapper to easily use your existing TorchScript models
with the minimal inference framework, maintaining compatibility with your test_TS.py workflow.
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Tuple, Optional

# Add the tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from minimal_inference import ImageProcessor, Visualizer, SegmentationInferencer


class TorchScriptModelWrapper(torch.nn.Module):
    """
    Wrapper for TorchScript models to work with the minimal inference framework.
    Provides a unified interface regardless of the TorchScript model's original signature.
    """
    
    def __init__(self, torchscript_model_path: str, device: str = 'cuda'):
        super().__init__()
        self.ts_model = torch.jit.load(torchscript_model_path, map_location=device)
        self.ts_model.eval()
        self.device = device
        
        # Try to determine the model's expected input signature
        self._analyze_model_signature()
    
    def _analyze_model_signature(self):
        """Analyze the TorchScript model to understand its input signature."""
        # Try to get the model's graph to understand inputs
        try:
            graph = self.ts_model.graph
            inputs = list(graph.inputs())
            print(f"TorchScript model expects {len(inputs)} inputs")
            
            # Check if it's a simple tensor input or needs additional parameters
            self.simple_input = len(inputs) <= 2  # Just model + tensor
            
        except Exception as e:
            print(f"Could not analyze model signature: {e}")
            self.simple_input = True
    
    def forward(self, inputs: torch.Tensor, img_metas: Optional[List[dict]] = None) -> torch.Tensor:
        """
        Unified forward method that works with different TorchScript model signatures.
        """
        if self.simple_input:
            # Simple case: model just takes tensor input
            return self.ts_model(inputs)
        else:
            # Complex case: model might need additional parameters
            if img_metas and len(img_metas) > 0:
                # Try to extract height and width from metadata
                height = img_metas[0]['img_shape'][0]
                width = img_metas[0]['img_shape'][1]
                
                # Try different signatures based on your model's interface
                try:
                    # Try with height, width parameters (for predict_torchscript method)
                    return self.ts_model(inputs, height, width)
                except:
                    # Fallback to simple input
                    return self.ts_model(inputs)
            else:
                # No metadata, use simple input
                return self.ts_model(inputs)


class CompatibleImageProcessor(ImageProcessor):
    """
    Image processor that's compatible with your existing test_TS.py preprocessing.
    """
    
    def __init__(self, input_shape: Tuple[int, int] = (512, 512), **kwargs):
        super().__init__(**kwargs)
        self.target_height, self.target_width = input_shape
    
    def preprocess(self, img_path: str, target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess image using the same logic as your test_TS.py load_image function.
        """
        import cv2
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original shape
        ori_h, ori_w = img.shape[:2]
        
        # Get current dimensions
        h, w = img.shape[:2]
        
        # Calculate target dimensions that are multiples of 8 (matching your logic)
        target_h = ((h + 7) // 8) * 8
        target_w = ((w + 7) // 8) * 8
        
        # Pad the image to make dimensions multiples of 8
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Pad symmetrically (top/bottom, left/right)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                cv2.BORDER_REFLECT_101)
        
        # Normalize (same as your test_TS.py)
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        
        # Create metadata
        img_meta = {
            'img_shape': (target_h, target_w),
            'ori_shape': (ori_h, ori_w),
            'pad_shape': (target_h, target_w),
            'padding_size': [pad_top, pad_bottom, pad_left, pad_right],
            'scale_factor': 1.0,
            'flip': False,
            'filename': os.path.basename(img_path)
        }
        
        return img_tensor, img_meta


class TorchScriptInferencer:
    """
    Simple inferencer specifically designed for TorchScript models,
    compatible with your existing workflow.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', 
                 input_shape: Tuple[int, int] = (512, 512),
                 num_classes: int = 3):
        
        self.device = device
        self.num_classes = num_classes
        
        # Load TorchScript model
        self.model = TorchScriptModelWrapper(model_path, device)
        
        # Create processor and visualizer
        self.processor = CompatibleImageProcessor(input_shape=input_shape)
        self.visualizer = Visualizer(
            class_names=['Background', 'Crop', 'Weed'][:num_classes]
        )
    
    def infer_single(self, img_path: str, show: bool = False, 
                    save_path: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Infer on a single image - compatible with your test_TS.py workflow.
        """
        import time
        
        # Preprocess
        img_tensor, img_meta = self.processor.preprocess(img_path)
        img_tensor = img_tensor.to(self.device)
        
        # Inference (matching your test_TS.py timing)
        start = time.time()
        with torch.no_grad():
            output = self.model(img_tensor, [img_meta])
            
            # Handle different output formats
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            # Resize output to match original image if needed
            if len(output.shape) == 4:  # [B, C, H, W]
                output = output.squeeze(0)  # Remove batch dimension
            
            seg_map = output.argmax(dim=0).cpu().numpy()
        
        end = time.time()
        inference_time = end - start
        
        # Postprocess - remove padding
        ori_h, ori_w = img_meta['ori_shape'][:2]
        pad_top, pad_bottom, pad_left, pad_right = img_meta['padding_size']
        
        if pad_bottom > 0:
            seg_map = seg_map[:-pad_bottom, :]
        if pad_right > 0:
            seg_map = seg_map[:, :-pad_right]
        if pad_top > 0:
            seg_map = seg_map[pad_top:, :]
        if pad_left > 0:
            seg_map = seg_map[:, pad_left:]
        
        # Resize to original size if needed
        if seg_map.shape[:2] != (ori_h, ori_w):
            import cv2
            seg_map = cv2.resize(seg_map, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        
        # Visualize if requested
        if show or save_path:
            self.visualizer.visualize(img_path, seg_map, save_path, show, alpha=0.5)
        
        return seg_map, inference_time
    
    def infer_folder(self, folder_path: str, output_dir: Optional[str] = None):
        """
        Process all images in a folder - compatible with your test_TS.py workflow.
        """
        import glob
        import os
        
        # Find all images
        extensions = ['jpg', 'png', 'jpeg']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext.upper()}')))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        total_time = 0
        for i, img_path in enumerate(image_files):
            save_path = None
            if output_dir:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(output_dir, f"{base_name}_seg.png")
            
            try:
                seg_map, inference_time = self.infer_single(img_path, save_path=save_path)
                total_time += inference_time
                fps = 1.0 / inference_time
                print(f"Image {i+1}/{len(image_files)}: {os.path.basename(img_path)} - {fps:.2f} FPS")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if len(image_files) > 0:
            avg_fps = len(image_files) / total_time
            print(f"Average FPS: {avg_fps:.2f}")


def main():
    """Main function - provides the same interface as your test_TS.py"""
    parser = argparse.ArgumentParser(description="TorchScript Segmentation Inference")
    parser.add_argument('model_path', help='Path to TorchScript model (.pt)')
    parser.add_argument('input_path', help='Path to input image, or folder')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[512, 512], 
                       help='Input shape (H W)')
    parser.add_argument('--mode', choices=['image', 'folder'], default='image', 
                       help='Input type')
    parser.add_argument('--output', type=str, help='Output directory for visualizations')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = TorchScriptInferencer(
        model_path=args.model_path,
        device=args.device,
        input_shape=tuple(args.input_shape),
        num_classes=args.num_classes
    )
    
    print(f"Loaded TorchScript model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Input shape: {args.input_shape}")
    
    # Run inference
    if args.mode == 'image':
        print(f"Processing single image: {args.input_path}")
        save_path = None
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.input_path))[0]
            save_path = os.path.join(args.output, f"{base_name}_seg.png")
        
        seg_map, inference_time = inferencer.infer_single(
            args.input_path, show=args.show, save_path=save_path)
        fps = 1.0 / inference_time
        print(f"Inference time: {inference_time:.3f}s ({fps:.1f} FPS)")
        
    elif args.mode == 'folder':
        print(f"Processing folder: {args.input_path}")
        inferencer.infer_folder(args.input_path, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()
