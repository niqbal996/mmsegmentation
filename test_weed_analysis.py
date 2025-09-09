#!/usr/bin/env python3
"""
Test script to demonstrate the weed-specific analysis functionality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from test_TS import compute_image_confidence, compute_weed_confidence, compute_class_specific_confidence

def create_sample_prob_map(scenario="balanced"):
    """Create sample probability maps for testing."""
    torch.manual_seed(42)  # For reproducible results
    
    if scenario == "balanced":
        # Balanced scenario: equal distribution of classes
        prob_map = torch.rand(3, 64, 64)
        prob_map = F.softmax(prob_map, dim=0)
        
    elif scenario == "high_weed_confidence":
        # High confidence weed predictions
        prob_map = torch.zeros(3, 64, 64)
        # Create some high-confidence weed regions
        prob_map[2, 20:40, 20:40] = 0.9  # High weed confidence
        prob_map[0, 20:40, 20:40] = 0.05  # Low background
        prob_map[1, 20:40, 20:40] = 0.05  # Low crop
        
        # Fill rest with background
        prob_map[0, :20, :] = 0.8
        prob_map[0, 40:, :] = 0.8
        prob_map[0, :, :20] = 0.8
        prob_map[0, :, 40:] = 0.8
        
        prob_map = F.softmax(prob_map, dim=0)
        
    elif scenario == "low_weed_confidence":
        # Low confidence weed predictions (uncertain)
        prob_map = torch.zeros(3, 64, 64)
        # Create some uncertain weed regions
        prob_map[2, 20:40, 20:40] = 0.4  # Low weed confidence
        prob_map[0, 20:40, 20:40] = 0.35  # Competing background
        prob_map[1, 20:40, 20:40] = 0.25  # Competing crop
        
        # Fill rest with background
        prob_map[0, :20, :] = 0.8
        prob_map[0, 40:, :] = 0.8
        prob_map[0, :, :20] = 0.8
        prob_map[0, :, 40:] = 0.8
        
        prob_map = F.softmax(prob_map, dim=0)
        
    elif scenario == "no_weeds":
        # No weeds detected
        prob_map = torch.zeros(3, 64, 64)
        prob_map[0, :32, :] = 0.8  # Background
        prob_map[1, 32:, :] = 0.8  # Crop
        prob_map = F.softmax(prob_map, dim=0)
        
    return prob_map

def analyze_scenario(scenario_name, prob_map):
    """Analyze a probability map scenario."""
    print(f"\n=== {scenario_name.upper()} SCENARIO ===")
    
    # Overall metrics
    overall_confidence = compute_image_confidence(prob_map)
    print(f"Overall Confidence: {overall_confidence:.4f}")
    
    # Weed-specific metrics
    weed_metrics = compute_weed_confidence(prob_map, weed_class_idx=2)
    print(f"Weed Metrics:")
    for key, value in weed_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Class-specific analysis for all classes
    class_names = ['Background', 'Crop', 'Weed']
    for class_idx in range(3):
        metrics = compute_class_specific_confidence(prob_map, class_idx)
        print(f"{class_names[class_idx]} Class Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

def main():
    print("Testing Weed-Specific Analysis Functions")
    print("=" * 50)
    
    scenarios = [
        "balanced",
        "high_weed_confidence", 
        "low_weed_confidence",
        "no_weeds"
    ]
    
    for scenario in scenarios:
        prob_map = create_sample_prob_map(scenario)
        analyze_scenario(scenario, prob_map)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("\nKey Insights:")
    print("1. avg_weed_prediction_confidence: Confidence specifically for pixels predicted as weed")
    print("2. weed_pixel_ratio: Proportion of image predicted as weed")
    print("3. avg_weed_max_confidence: Overall prediction confidence for weed regions")
    print("4. These metrics help identify:")
    print("   - Images with uncertain weed predictions (low confidence)")
    print("   - Images with high weed presence (high ratio)")
    print("   - Images where the model is very confident about weeds")

if __name__ == "__main__":
    main()
