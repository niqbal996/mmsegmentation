#!/usr/bin/env python3
"""
Test script for the new feature extraction functionality.
This script demonstrates how to use the FeatureExtractor class and related functions.
"""

import torch
import torch.nn as nn
import numpy as np
from test_TS import FeatureExtractor, compute_feature_entropy, extract_features_with_inference, print_model_layers, find_backbone_layers

# Create a simple test model to demonstrate the functionality
class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),  # 3 classes
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.decoder(features)
        return output

def test_feature_extraction():
    print("=== Testing Feature Extraction Functionality ===\n")
    
    # Create test model and input
    model = SimpleSegmentationModel()
    model.eval()
    
    # Create test input (batch_size=1, channels=3, height=256, width=256)
    test_input = torch.randn(1, 3, 256, 256)
    
    print("1. Model Structure Exploration:")
    print_model_layers(model, max_depth=3)
    
    print("\n2. Finding Backbone Layers:")
    backbone_layers = find_backbone_layers(model)
    print("Found backbone layers:")
    for layer in backbone_layers:
        print(f"  - {layer}")
    
    print("\n3. Manual Feature Extraction:")
    # Test individual feature extractor
    extractor = FeatureExtractor(model)
    
    # Register hooks for specific layers
    layer_names = ['backbone.0', 'backbone.3', 'backbone.6']  # Conv layers
    for layer_name in layer_names:
        extractor.register_hook(layer_name)
    
    # Run inference
    with torch.no_grad():
        output = model(test_input)
    
    # Get extracted features
    features = extractor.get_features()
    print("Extracted features:")
    for layer_name, feature_tensor in features.items():
        entropy = compute_feature_entropy(feature_tensor)
        print(f"  {layer_name}: {feature_tensor.shape} (entropy: {entropy:.4f})")
    
    # Clean up
    extractor.remove_hooks()
    
    print("\n4. All-in-One Feature Extraction:")
    # Test the convenience function
    output, features, entropies = extract_features_with_inference(
        model, test_input, layer_names, compute_entropy=True
    )
    
    print("Results from extract_features_with_inference:")
    print(f"  Output shape: {output.shape}")
    print("  Features:")
    for layer_name in layer_names:
        if layer_name in features:
            feat_shape = features[layer_name].shape
            entropy_val = entropies.get(layer_name, 0.0)
            print(f"    {layer_name}: {feat_shape} (entropy: {entropy_val:.4f})")
    
    print("\n5. Feature Entropy Analysis:")
    # Demonstrate different entropy calculations
    for layer_name, feature_tensor in features.items():
        entropy = compute_feature_entropy(feature_tensor)
        
        # Additional statistics
        mean_activation = torch.mean(feature_tensor).item()
        std_activation = torch.std(feature_tensor).item()
        
        print(f"  {layer_name}:")
        print(f"    Entropy: {entropy:.4f}")
        print(f"    Mean activation: {mean_activation:.4f}")
        print(f"    Std activation: {std_activation:.4f}")
    
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_feature_extraction()
