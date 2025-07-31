#!/usr/bin/env python3
"""
Crop/Weed Dataset Visualizer

This script provides an interactive visualization tool for semantic segmentation datasets
with images and their corresponding ground truth masks.

Usage:
    python cropweed_visualizer.py /path/to/dataset --subset Sugarbeet1
    
Navigation:
    - Press 'n' or → for next image
    - Press 'p' or ← for previous image
    - Press 'q' or ESC to quit
    - Press 's' to save current visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path


class CropWeedVisualizer:
    def __init__(self, root_path, subset_type=None, image_extensions=('*.png', '*.jpg', '*.jpeg')):
        """
        Initialize the visualizer.
        
        Args:
            root_path (str): Root directory path containing 'images' and 'labelIds' folders
            subset_type (str, optional): Subset type (e.g., 'Sugarbeet1') for mask filtering
            image_extensions (tuple): Supported image file extensions
        """
        self.root_path = Path(root_path)
        self.subset_type = subset_type
        self.image_extensions = image_extensions
        
        # Define consistent color palette for classes
        self.color_palette = {
            0: [0, 255, 0],        # Crop - Green  
            1: [255, 0, 0],      # Weed - Red  
            2: [0, 0, 0],      # Background - Black
            3: [0, 0, 255],      # Other - Blue
            4: [255, 255, 0],    # Additional class - Yellow
            5: [255, 0, 255],    # Additional class - Magenta
            6: [0, 255, 255],    # Additional class - Cyan
        }
        
        self.class_names = {
            0: 'Background',
            1: 'Crop',
            2: 'Weed', 
            3: 'Other',
            4: 'Class 4',
            5: 'Class 5',
            6: 'Class 6',
        }
        
        # Load image and mask file lists
        self.image_files, self.mask_files = self._load_file_lists()
        self.current_idx = 0
        
        if not self.mask_files:
            raise ValueError(f"No masks found in {self.root_path / 'labelIds'}")
        
        print(f"Found {len(self.mask_files)} mask files")
        if subset_type:
            print(f"Using subset: {subset_type}")
        
        # Initialize matplotlib
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Display first image
        self._display_current()
    
    def _load_file_lists(self):
        """Load mask file list (images will be derived from mask paths)."""
        masks_dir = self.root_path / 'labelIds'
        
        # Get all mask files first (since they are fewer)
        mask_files = []
        
        if self.subset_type:
            # Look in specific subset folder
            subset_mask_dir = masks_dir / self.subset_type
            if subset_mask_dir.exists():
                for mask_ext in ['*.png', '*.jpg', '*.jpeg', '*.npz']:
                    mask_files.extend(glob.glob(str(subset_mask_dir / mask_ext)))
        else:
            # Search in all subdirectories of labelIds
            for mask_ext in ['*.png', '*.jpg', '*.jpeg', '*.npz']:
                mask_files.extend(glob.glob(str(masks_dir / '**' / mask_ext), recursive=True))
        
        mask_files = sorted(mask_files)
        
        if not mask_files:
            return [], []
        
        # Return mask files twice - we'll derive image paths when needed
        return mask_files, mask_files
    
    def _get_image_path_from_mask(self, mask_path):
        """Convert mask path to corresponding image path."""
        mask_path = Path(mask_path)
        mask_name = mask_path.stem  # filename without extension
        images_dir = self.root_path / 'images'
        
        # Try different image extensions
        for img_ext in self.image_extensions:
            ext = img_ext.replace('*', '')
            img_path = images_dir / f"{mask_name}{ext}"
            if img_path.exists():
                return str(img_path)
        
        # If not found in root images dir, try recursive search (fallback)
        for img_ext in self.image_extensions:
            ext = img_ext.replace('*', '')
            img_pattern = str(images_dir / '**' / f"{mask_name}{ext}")
            img_files = glob.glob(img_pattern, recursive=True)
            if img_files:
                return img_files[0]
        
        # Return a placeholder if not found
        return str(images_dir / f"{mask_name}.png")
    
    def _load_mask(self, mask_path):
        """Load mask from file (supports .png, .jpg, .npz)."""
        mask_path = Path(mask_path)
        
        if mask_path.suffix.lower() == '.npz':
            # Load from npz file
            data = np.load(mask_path)
            if 'array' in data:
                mask = data['array']
            else:
                # Take the first array in the npz file
                mask = data[list(data.keys())[0]]
        else:
            # Load regular image file
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        return mask
    
    def _create_colored_mask(self, mask):
        """Convert grayscale mask to colored visualization."""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            if class_id in self.color_palette:
                color = self.color_palette[class_id]
                colored_mask[mask == class_id] = color
            else:
                # Use a default color for unknown classes
                colored_mask[mask == class_id] = [128, 128, 128]  # Gray
        
        return colored_mask
    
    def _display_current(self):
        """Display current image and mask pair."""
        if not self.mask_files:
            return
        
        # Load current mask and derive image path
        mask_path = self.mask_files[self.current_idx]
        img_path = self._get_image_path_from_mask(mask_path)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = self._load_mask(mask_path)
        colored_mask = self._create_colored_mask(mask)
        
        # Create overlay
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Display original image
        self.axes[0].imshow(image)
        self.axes[0].set_title(f"Original Image\n{Path(img_path).name}")
        self.axes[0].axis('off')
        
        # Display colored mask
        self.axes[1].imshow(colored_mask)
        self.axes[1].set_title(f"Ground Truth Mask\n{Path(mask_path).name}")
        self.axes[1].axis('off')
        
        # Display overlay
        self.axes[2].imshow(overlay)
        self.axes[2].set_title("Overlay")
        self.axes[2].axis('off')
        
        # Update main title with navigation info
        unique_classes = np.unique(mask)
        class_info = [f"{self.class_names.get(c, f'Class {c}')}" for c in unique_classes if c in self.class_names]
        
        self.fig.suptitle(
            f"Image {self.current_idx + 1}/{len(self.mask_files)} | "
            f"Classes: {', '.join(class_info)} | "
            f"Subset: {self.subset_type or 'All'}",
            fontsize=14, fontweight='bold'
        )
        
        # Add legend
        self._add_legend(unique_classes)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def _add_legend(self, unique_classes):
        """Add color legend for classes present in current image."""
        legend_elements = []
        for class_id in sorted(unique_classes):
            if class_id in self.color_palette:
                color = np.array(self.color_palette[class_id]) / 255.0
                label = self.class_names.get(class_id, f'Class {class_id}')
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=label))
        
        if legend_elements:
            self.fig.legend(handles=legend_elements, loc='lower center', 
                          bbox_to_anchor=(0.5, -0.05), ncol=len(legend_elements))
    
    def _on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key in ['n', 'right']:
            self.next_image()
        elif event.key in ['p', 'left']:
            self.prev_image()
        elif event.key in ['q', 'escape']:
            self.quit()
        elif event.key == 's':
            self.save_current()
        elif event.key == 'h':
            self.show_help()
    
    def next_image(self):
        """Navigate to next image."""
        self.current_idx = (self.current_idx + 1) % len(self.mask_files)
        self._display_current()
    
    def prev_image(self):
        """Navigate to previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.mask_files)
        self._display_current()
    
    def save_current(self):
        """Save current visualization."""
        output_dir = Path('./visualizations')
        output_dir.mkdir(exist_ok=True)
        
        mask_path = self.mask_files[self.current_idx]
        img_name = Path(mask_path).stem
        save_path = output_dir / f"{img_name}_visualization.png"
        
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    def show_help(self):
        """Display help information."""
        help_text = """
        Navigation Help:
        ---------------
        'n' or →     : Next image
        'p' or ←     : Previous image
        's'          : Save current visualization
        'h'          : Show this help
        'q' or ESC   : Quit
        
        Make sure the matplotlib window has focus to register keypresses.
        """
        print(help_text)
    
    def quit(self):
        """Quit the visualizer."""
        plt.close('all')
        plt.ioff()
        print("Visualizer closed.")
    
    def run(self):
        """Start the interactive visualization."""
        print("\nCrop/Weed Dataset Visualizer")
        print("=" * 40)
        self.show_help()
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            self.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Crop/Weed Dataset Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cropweed_visualizer.py /path/to/dataset
    python cropweed_visualizer.py /path/to/dataset --subset Sugarbeet1
    python cropweed_visualizer.py /path/to/dataset --subset Sugarbeet1 --extensions png jpg
        """
    )
    
    parser.add_argument('root_path', 
                       help='Root directory containing "images" and "labelIds" folders')
    parser.add_argument('--subset', 
                       help='Subset type (e.g., Sugarbeet1) for filtering masks')
    parser.add_argument('--extensions', nargs='+', default=['png', 'jpg', 'jpeg'],
                       help='Image file extensions to look for (default: png jpg jpeg)')
    
    args = parser.parse_args()
    
    # Validate root path
    root_path = Path(args.root_path)
    if not root_path.exists():
        print(f"Error: Root path '{root_path}' does not exist.")
        return
    
    if not (root_path / 'images').exists():
        print(f"Error: 'images' folder not found in '{root_path}'")
        return
    
    if not (root_path / 'labelIds').exists():
        print(f"Error: 'labelIds' folder not found in '{root_path}'")
        return
    
    # Prepare extensions
    extensions = [f'*.{ext}' for ext in args.extensions]
    
    try:
        # Create and run visualizer
        visualizer = CropWeedVisualizer(
            root_path=args.root_path,
            subset_type=args.subset,
            image_extensions=extensions
        )
        visualizer.run()
    
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == '__main__':
    main()
