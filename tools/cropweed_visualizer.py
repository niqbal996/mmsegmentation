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


class Dataset:
    """Simple dataset class to hold class information."""
    def __init__(self, class_dict):
        self.class_dict = class_dict


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
        
        # Define dataset mappings
        self.DATASETS = {}
        
        # CropAndWeed dataset mapping
        self.DATASETS['CropAndWeed'] = Dataset({
            0: ('Soil', (0, 0, 0)),
            1: ('Maize', (255, 0, 0)),
            2: ('Maize two-leaf stage', (234, 0, 0)),
            3: ('Maize four-leaf stage', (212, 0, 0)),
            4: ('Maize six-leaf stage', (191, 0, 0)),
            5: ('Maize eight-leaf stage', (170, 0, 0)),
            6: ('Maize max', (149, 0, 0)),
            7: ('Sugar beet', (255, 85, 0)),
            8: ('Sugar beet two-leaf stage', (234, 78, 0)),
            9: ('Sugar beet four-leaf stage', (212, 71, 0)),
            10: ('Sugar beet six-leaf stage', (191, 64, 0)),
            11: ('Sugar beet eight-leaf stage', (170, 57, 0)),
            12: ('Sugar beet Max', (149, 50, 0)),
            13: ('Pea', (255, 170, 0)),
            14: ('Courgette', (255, 255, 0)),
            15: ('Pumpkins', (170, 255, 0)),
            16: ('Radish', (85, 255, 0)),
            17: ('Asparagus', (0, 255, 0)),
            18: ('Potato', (0, 255, 85)),
            19: ('Flat leaf parsley', (0, 255, 170)),
            20: ('Curly leaf parsley', (0, 255, 255)),
            21: ('Cowslip', (0, 170, 255)),
            22: ('Poppy', (0, 85, 255)),
            23: ('Hemp', (0, 0, 255)),
            24: ('Sunflower', (85, 0, 255)),
            25: ('Sage', (170, 0, 255)),
            26: ('Common bean', (255, 0, 255)),
            27: ('Faba bean', (255, 0, 170)),
            28: ('Clover', (255, 0, 85)),
            29: ('Hybrid goosefoot', (255, 188, 178)),
            30: ('Black-bindweed', (255, 207, 178)),
            31: ('Cockspur grass', (255, 226, 178)),
            32: ('Red-root amaranth', (255, 245, 178)),
            33: ('White goosefoot', (245, 255, 178)),
            34: ('Thorn apple', (226, 255, 178)),
            35: ('Potato weed', (207, 255, 178)),
            36: ('German chamomile', (188, 255, 178)),
            37: ('Saltbush', (178, 255, 188)),
            38: ('Creeping thistle', (178, 255, 207)),
            39: ('Field milk thistle', (178, 255, 226)),
            40: ('Purslane', (178, 255, 245)),
            41: ('Black nightshade', (178, 245, 255)),
            42: ('Mercuries', (178, 226, 255)),
            43: ('Spurge', (178, 207, 255)),
            44: ('Pale persicaria', (178, 188, 255)),
            45: ('Geraniums', (188, 178, 255)),
            46: ('Cleavers', (207, 178, 255)),
            47: ('Whitetop', (226, 178, 255)),
            48: ('Meadow-grass', (245, 178, 255)),
            49: ('Frosted orach', (255, 178, 245)),
            50: ('Black horehound', (255, 178, 226)),
            51: ('Shepherds purse', (255, 178, 207)),
            52: ('Field bindweed', (255, 178, 188)),
            53: ('Common mugwort', (255, 194, 178)),
            54: ('Hedge mustard', (255, 213, 178)),
            55: ('Groundsel', (255, 219, 178)),
            56: ('Speedwell', (255, 232, 178)),
            57: ('Broadleaf plantain', (255, 238, 178)),
            58: ('White ball-mustard', (255, 251, 178)),
            59: ('Peppermint', (255, 212, 0)),
            60: ('Field pennycress', (239, 255, 178)),
            61: ('Corn spurry', (233, 255, 178)),
            62: ('Purple crabgrass', (220, 255, 178)),
            63: ('Common fumitory', (214, 255, 178)),
            64: ('Ivy-leaved speedwell', (201, 255, 178)),
            65: ('Annual meadow grass', (195, 255, 178)),
            66: ('Redshank', (182, 255, 178)),
            67: ('Common hemp-nettle', (178, 255, 194)),
            68: ('Rough meadow-grass', (178, 255, 200)),
            69: ('Green bristlegrass', (178, 255, 213)),
            70: ('Small geranium', (178, 255, 220)),
            71: ('Cornflower', (178, 255, 232)),
            72: ('Common corn-cockle', (178, 255, 238)),
            73: ('Creeping crowfoot', (178, 255, 251)),
            74: ('Wall barley', (178, 239, 255)),
            75: ('Annual fescue', (178, 233, 255)),
            76: ('Purple dead-nettle', (178, 220, 255)),
            77: ('Ribwort plantain', (178, 214, 255)),
            78: ('Pineappleweed', (178, 201, 255)),
            79: ('Common chickweed', (178, 195, 255)),
            80: ('Hedge mustard', (178, 182, 255)),
            81: ('Soft brome', (194, 178, 255)),
            82: ('Wild pansy', (200, 178, 255)),
            83: ('Yellow rocket', (213, 178, 255)),
            84: ('Common wild oat', (219, 178, 255)),
            85: ('Red poppy', (232, 178, 255)),
            86: ('Rye brome', (238, 178, 255)),
            87: ('Knotgrass', (251, 178, 255)),
            88: ('Prickly lettuce', (255, 178, 239)),
            89: ('Copse-bindweed', (255, 178, 233)),
            90: ('Manyseeds', (255, 178, 220)),
            91: ('Common buckwheat', (255, 178, 214)),
            92: ('Chives', (212, 255, 0)),
            93: ('Garlic', (127, 255, 0)),
            94: ('Soybean', (42, 255, 0)),
            95: ('Wild carrot', (244, 255, 0)),
            96: ('Field mustard', (159, 255, 0)),
            97: ('Giant fennel', (74, 255, 0)),
            98: ('Common horsetail', (10, 255, 0)),
            99: ('Common dandelion', (202, 255, 0)),
            255: ('Vegetation', (128, 128, 128))
        })
        
        # SugarBeetFine dataset mapping
        self.DATASETS['SugarBeetFine'] = Dataset({
            0: ('Sugar beet', (255, 85, 0)),
            1: ('Amaranth', (255, 245, 178)),
            2: ('Grasses', (255, 226, 178)),
            3: ('Goosefoot', (226, 255, 178)),
            4: ('Knotweed', (255, 207, 178)),
            5: ('Corn spurry', (233, 255, 178)),
            6: ('Chickweed', (178, 195, 255)),
            7: ('Solanales', (226, 255, 178)),
            8: ('Potato weed', (207, 255, 178)),
            9: ('Chamomile', (188, 255, 178)),
            10: ('Thistle', (178, 255, 207)),
            11: ('Mercuries', (178, 226, 255)),
            12: ('Geranium', (188, 178, 255)),
            13: ('Crucifer', (239, 255, 178)),
            14: ('Poppy', (214, 255, 178)),
            15: ('Plantago', (255, 232, 178)),
            16: ('Labiate', (255, 212, 0))
        })
        
        # Simple 3-class dataset mapping
        self.DATASETS['Simple3Class'] = Dataset({
            0: ('Background', (0, 0, 0)),
            1: ('Crop', (0, 255, 0)),
            2: ('Weed', (255, 0, 0))
        })
        
        # Load image and mask file lists first
        self.image_files, self.mask_files = self._load_file_lists()
        
        # Then determine which dataset to use based on subset_type and actual data
        self.current_dataset = self._select_dataset()
        self.color_palette, self.class_names = self._build_color_palette()
        
        self.current_idx = 0
        
        if not self.mask_files:
            raise ValueError(f"No masks found in {self.root_path / 'labelIds'}")
        
        print(f"Found {len(self.mask_files)} mask files")
        print(f"Using dataset: {self.current_dataset}")
        if subset_type:
            print(f"Using subset: {subset_type}")
        
        # Initialize matplotlib
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Display first image
        self._display_current()
    
    def _select_dataset(self):
        """Select appropriate dataset based on subset_type."""
        if self.subset_type:
            subset_lower = self.subset_type.lower()
            if 'cropandweed' in subset_lower or 'crop_and_weed' in subset_lower:
                return 'CropAndWeed'
            elif 'sugarbeet' in subset_lower or 'sugar_beet' in subset_lower:
                return 'SugarBeetFine'
            elif 'phenobench' in subset_lower or '3class' in subset_lower:
                return 'Simple3Class'
        
        # Default fallback - try to detect from available classes in first mask
        try:
            if self.mask_files:
                first_mask = self._load_mask(self.mask_files[0])
                unique_classes = np.unique(first_mask)
                max_class = np.max(unique_classes)
                
                if max_class > 50:  # Likely CropAndWeed with many classes
                    return 'CropAndWeed'
                elif max_class > 16:  # Likely SugarBeetFine
                    return 'SugarBeetFine'
                else:  # Simple 3-class
                    return 'Simple3Class'
        except:
            pass
        
        return 'Simple3Class'  # Final fallback
    
    def _build_color_palette(self):
        """Build color palette and class names from selected dataset."""
        dataset = self.DATASETS[self.current_dataset]
        color_palette = {}
        class_names = {}
        
        for class_id, (class_name, color) in dataset.class_dict.items():
            # Ensure background (class 0) is always black
            if class_id == 0:
                color_palette[class_id] = [0, 0, 0]
            else:
                color_palette[class_id] = list(color)
            class_names[class_id] = class_name
        
        return color_palette, class_names
    
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
    
    def _create_colored_mask_with_labels(self, mask):
        """Convert grayscale mask to colored visualization with text labels."""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        unique_classes = np.unique(mask)
        class_regions = {}  # Store regions for each class for text placement
        
        for class_id in unique_classes:
            if class_id in self.color_palette:
                color = self.color_palette[class_id]
                colored_mask[mask == class_id] = color
                
                # Find region center for text placement
                class_mask = (mask == class_id)
                if np.sum(class_mask) > 100:  # Only add text for reasonably sized regions
                    y_coords, x_coords = np.where(class_mask)
                    center_y, center_x = np.mean(y_coords), np.mean(x_coords)
                    class_regions[class_id] = (int(center_x), int(center_y))
            else:
                # Use a default color for unknown classes
                colored_mask[mask == class_id] = [128, 128, 128]  # Gray
        
        return colored_mask, class_regions

    def _add_text_labels_to_mask(self, ax, mask, class_regions):
        """Add class name text labels to mask regions."""
        for class_id, (center_x, center_y) in class_regions.items():
            class_name = self.class_names.get(class_id, f'Class {class_id}')
            
            # Choose text color based on background
            if class_id in self.color_palette:
                bg_color = np.array(self.color_palette[class_id])
                # Use white text on dark backgrounds, black on light
                text_color = 'white' if np.mean(bg_color) < 128 else 'black'
            else:
                text_color = 'white'
            
            # Add text with background box for better readability
            ax.text(center_x, center_y, class_name, 
                   color=text_color, fontsize=8, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

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
        colored_mask, class_regions = self._create_colored_mask_with_labels(mask)
        
        # Create overlay
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Display original image
        self.axes[0].imshow(image)
        self.axes[0].set_title(f"Original Image\n{Path(img_path).name}")
        self.axes[0].axis('off')
        
        # Display colored mask with labels
        self.axes[1].imshow(colored_mask)
        self.axes[1].set_title(f"Ground Truth Mask\n{Path(mask_path).name}")
        self.axes[1].axis('off')
        
        # Add text labels to mask
        self._add_text_labels_to_mask(self.axes[1], mask, class_regions)
        
        # Display overlay
        self.axes[2].imshow(overlay)
        self.axes[2].set_title("Overlay")
        self.axes[2].axis('off')
        
        # Update main title with navigation info
        unique_classes = np.unique(mask)
        class_info = []
        class_counts = {}
        
        # Get class counts and names
        for c in unique_classes:
            if c in self.class_names:
                class_name = self.class_names[c]
                pixel_count = np.sum(mask == c)
                total_pixels = mask.size
                percentage = (pixel_count / total_pixels) * 100
                class_counts[c] = percentage
                if percentage > 1.0:  # Only show classes with >1% coverage
                    class_info.append(f"{class_name} ({percentage:.1f}%)")
        
        # Limit displayed classes to avoid overcrowding
        if len(class_info) > 6:
            # Show top 5 classes by percentage + "and X more"
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            top_classes = [self.class_names.get(c, f'Class {c}') for c, _ in sorted_classes[:5]]
            class_info = top_classes + [f"and {len(class_info)-5} more"]
        
        self.fig.suptitle(
            f"Image {self.current_idx + 1}/{len(self.mask_files)} | "
            f"Dataset: {self.current_dataset} | "
            f"Subset: {self.subset_type or 'All'} | "
            f"Classes: {len(unique_classes)}",
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
        max_legend_items = 8  # Limit legend items to avoid overcrowding
        
        # Sort classes by ID for consistent display
        sorted_classes = sorted(unique_classes)
        
        for i, class_id in enumerate(sorted_classes):
            if i >= max_legend_items:
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='gray', 
                                                   label=f'... and {len(sorted_classes) - i} more'))
                break
                
            if class_id in self.color_palette:
                color = np.array(self.color_palette[class_id]) / 255.0
                label = self.class_names.get(class_id, f'Class {class_id}')
                # Truncate long labels
                if len(label) > 15:
                    label = label[:12] + '...'
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=f'{class_id}: {label}'))
        
        if legend_elements:
            # Use multiple rows if needed
            ncol = min(len(legend_elements), 4)
            self.fig.legend(handles=legend_elements, loc='lower center', 
                          bbox_to_anchor=(0.5, -0.08), ncol=ncol, fontsize=9)
    
    def add_custom_dataset(self, dataset_name, class_dict):
        """Add a custom dataset mapping.
        
        Args:
            dataset_name (str): Name of the dataset
            class_dict (dict): Dictionary mapping class_id -> (class_name, color_tuple)
        """
        self.DATASETS[dataset_name] = Dataset(class_dict)
        print(f"Added custom dataset: {dataset_name}")
    
    def switch_dataset(self, dataset_name):
        """Switch to a different dataset mapping.
        
        Args:
            dataset_name (str): Name of the dataset to switch to
        """
        if dataset_name in self.DATASETS:
            self.current_dataset = dataset_name
            self.color_palette, self.class_names = self._build_color_palette()
            self._display_current()  # Refresh display
            print(f"Switched to dataset: {dataset_name}")
        else:
            print(f"Dataset '{dataset_name}' not found. Available: {list(self.DATASETS.keys())}")
    
    def list_datasets(self):
        """List all available datasets."""
        print("Available datasets:")
        for name, dataset in self.DATASETS.items():
            num_classes = len(dataset.class_dict)
            current = " (current)" if name == self.current_dataset else ""
            print(f"  - {name}: {num_classes} classes{current}")
    
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
        elif event.key == 'd':
            self.list_datasets()
        elif event.key == '1':
            self.switch_dataset('Simple3Class')
        elif event.key == '2':
            self.switch_dataset('SugarBeetFine')
        elif event.key == '3':
            self.switch_dataset('CropAndWeed')
        elif event.key == 'r':
            self._display_current()  # Refresh display
    
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
        help_text = f"""
        Crop/Weed Dataset Visualizer - Help
        ===================================
        
        Navigation:
        -----------
        'n' or →     : Next image
        'p' or ←     : Previous image
        'r'          : Refresh current display
        's'          : Save current visualization
        'q' or ESC   : Quit
        'h'          : Show this help
        
        Dataset Controls:
        ----------------
        'd'          : List available datasets
        '1'          : Switch to Simple3Class dataset
        '2'          : Switch to SugarBeetFine dataset  
        '3'          : Switch to CropAndWeed dataset
        
        Current Settings:
        ----------------
        Dataset: {self.current_dataset}
        Subset: {self.subset_type or 'All'}
        Images: {len(self.mask_files)}
        
        Features:
        ---------
        - Class names are displayed on mask regions
        - Background (class 0) is always black
        - Legend shows present classes with percentages
        - Colors are automatically assigned per dataset
        
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
    # Basic usage
    python cropweed_visualizer.py /path/to/dataset
    
    # With specific subset
    python cropweed_visualizer.py /path/to/dataset --subset Sugarbeet1 --dataset SugarBeetFine
    
    # CropAndWeed dataset with custom subset
    python cropweed_visualizer.py /path/to/dataset --subset CropAndWeed --dataset CropAndWeed
    
    # Custom file extensions
    python cropweed_visualizer.py /path/to/dataset --extensions png jpg npz

Dataset Types:
    - Simple3Class: Background, Crop, Weed (3 classes)
    - SugarBeetFine: Sugar beet with 17 fine-grained weed classes
    - CropAndWeed: Full CropAndWeed dataset with 100 classes

Navigation Keys:
    n/→: Next, p/←: Previous, s: Save, h: Help, q: Quit
    d: List datasets, 1/2/3: Switch datasets, r: Refresh
        """
    )
    
    parser.add_argument('root_path', 
                       help='Root directory containing "images" and "labelIds" folders')
    parser.add_argument('--subset', 
                       help='Subset type (e.g., Sugarbeet1, CropAndWeed) for filtering masks')
    parser.add_argument('--dataset', choices=['Simple3Class', 'SugarBeetFine', 'CropAndWeed'],
                       help='Force specific dataset mapping (auto-detected if not specified)')
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
        # Create visualizer
        visualizer = CropWeedVisualizer(
            root_path=args.root_path,
            subset_type=args.subset,
            image_extensions=extensions
        )
        
        # Force specific dataset if requested
        if args.dataset:
            visualizer.switch_dataset(args.dataset)
        
        # Show initial info
        print(f"\nDataset Information:")
        print(f"Root path: {args.root_path}")
        print(f"Subset: {args.subset or 'All'}")
        print(f"Current dataset: {visualizer.current_dataset}")
        print(f"Total images: {len(visualizer.mask_files)}")
        visualizer.list_datasets()
        
        # Run visualizer
        visualizer.run()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
