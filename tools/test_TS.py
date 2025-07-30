import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as transF
import os
import glob
import time
from tqdm import tqdm

def load_image(img_path, input_shape):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get current dimensions
    h, w = img.shape[:2]
    
    # Calculate target dimensions that are multiples of 8
    target_h = ((h + 7) // 8) * 8  # Round up to nearest multiple of 8
    target_w = ((w + 7) // 8) * 8  # Round up to nearest multiple of 8
    
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
    
    img = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img

def visualize_segmentation(img, seg_map, num_classes=3, show=True, save_path=None):
    seg_map = seg_map.squeeze()
    palette = np.array([
        [0, 0, 0],        # class 0: black
        [0, 255, 0],      # class 1: green
        [255, 0, 0],      # class 2: blue
    ], dtype=np.uint8)
    color_seg = palette[seg_map]
    color_seg = cv2.resize(color_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(img, 0.5, color_seg, 0.5, 0)
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Segmentation Overlay")
        plt.imshow(overlay)
        plt.axis('off')
        plt.show(block=True)
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay

def visualize_entropy_results(model, sorted_list, input_shape, n=5, num_classes=3):
    """Visualize highest and lowest entropy images with segmentation masks."""
    if len(sorted_list) < n:
        n = len(sorted_list)
    
    class_names = ['Background', 'Crop', 'Weed']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get highest entropy images (first n)
    highest_entropy = sorted_list[:n]
    # Get lowest entropy images (last n)
    lowest_entropy = sorted_list[-n:]
    
    all_images = []
    for category, images in [("Highest", highest_entropy), ("Lowest", lowest_entropy)]:
        for idx, (img_path, entropy_val, confidence, weed_confidence) in enumerate(images):
            all_images.append((category, idx, img_path, entropy_val, confidence, weed_confidence))

    current_idx = [0]  # Use list to make it mutable in nested function
    
    def show_current_image():
        category, idx_in_category, img_path, entropy_val, confidence, weed_confidence = all_images[current_idx[0]]

        # Load and process image
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = load_image(img_path, input_shape).to(device)
        
        # Get segmentation
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output_resized = torch.nn.functional.interpolate(
                output, size=(rgb_img.shape[0], rgb_img.shape[1]), mode='bilinear', align_corners=True
            )
            seg_map = output_resized.squeeze(0).argmax(dim=0).cpu().numpy()
        
        # Clear and recreate the plot
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        
        # Create subplots - just 1 row, 2 columns
        fig.suptitle(f"{category} Entropy Image {idx_in_category+1} ({current_idx[0]+1}/{len(all_images)})\n"
                    f"Entropy: {entropy_val:.4f} | Crop Confidence: {confidence:.4f} | Weed Confidence: {weed_confidence:.4f}",
                    fontsize=14, fontweight='bold')
        
        # Original image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(rgb_img)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Segmentation map only
        palette = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
        color_seg = palette[seg_map]
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(color_seg)
        ax2.set_title("Segmentation Map")
        ax2.axis('off')
        
        # Save the visualization
        save_path = f"{category.lower()}_entropy_{idx_in_category+1}_entropy_{entropy_val:.4f}.png"
        save_path = os.path.join('cam_output/GN1920', save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
        
        plt.draw()
        plt.pause(0.01)  # Small pause to ensure the plot updates
    
    def on_key(event):
        if event.key == 'n' or event.key == 'right':  # Next image
            current_idx[0] = (current_idx[0] + 1) % len(all_images)
            show_current_image()
        elif event.key == 'p' or event.key == 'left':  # Previous image
            current_idx[0] = (current_idx[0] - 1) % len(all_images)
            show_current_image()
        elif event.key == 'q' or event.key == 'escape':  # Quit
            plt.close('all')
            return
    
    # Create figure and set up
    plt.ion()  # Turn on interactive mode
    fig, _ = plt.subplots(figsize=(12, 6))
    
    # Connect the key press event
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show first image
    show_current_image()
    
    print("\nNavigation:")
    print("- Press 'n' or → for next image")
    print("- Press 'p' or ← for previous image") 
    print("- Press 'q' or ESC to quit")
    print("- Make sure the matplotlib window has focus to register keypresses")
    
    # Keep the plot open and responsive
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        plt.close('all')
    finally:
        fig.canvas.mpl_disconnect(cid)
        plt.ioff()  # Turn off interactive mode

def infer_and_time(model, img_tensor, input_img, num_classes=3, show=False, save_path=None):
    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output_resized = torch.nn.functional.interpolate(
            output, size=(1024, 1024), mode='bilinear', align_corners=True
        )
        output_resized = output_resized.squeeze(0)
        seg_map = output_resized.argmax(dim=0, keepdim=True)
        seg_map = seg_map.cpu().numpy()
    end = time.time()
    overlay = visualize_segmentation(input_img, seg_map, num_classes=num_classes, show=show, save_path=save_path)
    return end - start, overlay

def infer_on_video(model, video_path, input_shape, num_classes=3):
    cap = cv2.VideoCapture(video_path)
    frame_times = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get current dimensions
        h, w = rgb_frame.shape[:2]
        
        # Calculate target dimensions that are multiples of 8
        target_h = ((h + 7) // 8) * 8
        target_w = ((w + 7) // 8) * 8
        
        # Pad the image to make dimensions multiples of 8
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img = cv2.copyMakeBorder(rgb_frame, pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_REFLECT_101)
        
        img_tensor = img.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()
        t, _ = infer_and_time(model, img_tensor, rgb_frame, num_classes=num_classes, show=False)
        frame_times.append(t)
        frame_count += 1
        print(f"Frame {frame_count}: {1.0/t:.2f} FPS")
    cap.release()
    if frame_times:
        print(f"Average FPS: {1.0/np.mean(frame_times):.2f}")

def infer_on_folder(model, folder_path, input_shape, num_classes=3):
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.[jp][pn]g')))
    if not image_files:
        print("No images found in folder.")
        return
    frame_times = []
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = load_image(img_path, input_shape)
        t, _ = infer_and_time(model, img_tensor, rgb_img, num_classes=num_classes, show=False)
        frame_times.append(t)
        print(f"Image {idx+1}/{len(image_files)}: {1.0/t:.2f} FPS")
    if frame_times:
        print(f"Average FPS: {1.0/np.mean(frame_times):.2f}")

def entropy_image_avg(pred):
    b = F.softmax(pred, dim=0)* F.log_softmax(pred, dim=0)
    b = -1.0 * b.sum(dim=0)
    b = (255*(b - b.min())/(b.max()-b.min()))
    
    c = b.cpu().numpy().astype(np.uint8)
    out = cv2.applyColorMap(c, cv2.COLORMAP_JET)
    return out, torch.sum(b).cpu().numpy()/(b.size(0)*b.size(1))

def compute_image_entropy(prob_map):
    """
    Compute the image-level entropy from per-pixel probability maps.

    Args:
        prob_map (Tensor): shape (C, H, W), softmax probabilities over classes.

    Returns:
        entropy (float): average entropy over all pixels.
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    entropy_map = -torch.sum(prob_map * torch.log(prob_map + eps), dim=0)  # shape (H, W)
    avg_entropy = torch.mean(entropy_map).item()
    return avg_entropy

def compute_image_confidence(prob_map):
    """
    Compute the image-level confidence from per-pixel probability maps.

    Args:
        prob_map (Tensor): shape (C, H, W), softmax probabilities over classes.

    Returns:
        confidence (float): average max probability over all pixels.
    """
    max_probs, _ = torch.max(prob_map, dim=0)  # shape (H, W)
    avg_confidence = torch.mean(max_probs).item()
    return avg_confidence

def compute_weed_confidence(prob_map, weed_class_idx=2):
    """
    Compute weed-specific confidence metrics.

    Args:
        prob_map (Tensor): shape (C, H, W), softmax probabilities over classes.
        weed_class_idx (int): Index of the weed class (default: 2).

    Returns:
        dict: Dictionary containing various weed-related confidence metrics.
    """
    # Get weed probability map
    weed_probs = prob_map[weed_class_idx]  # shape (H, W)
    
    # Get predicted class for each pixel
    pred_classes = torch.argmax(prob_map, dim=0)  # shape (H, W)
    
    # Find pixels predicted as weed
    weed_mask = (pred_classes == weed_class_idx)
    
    metrics = {}
    
    # Overall weed probability statistics
    metrics['avg_weed_prob'] = torch.mean(weed_probs).item()
    metrics['max_weed_prob'] = torch.max(weed_probs).item()
    metrics['min_weed_prob'] = torch.min(weed_probs).item()
    
    # Weed detection metrics
    total_pixels = pred_classes.numel()
    weed_pixels = torch.sum(weed_mask).item()
    metrics['weed_pixel_ratio'] = weed_pixels / total_pixels
    
    # Confidence of weed predictions (only for pixels predicted as weed)
    if weed_pixels > 0:
        weed_prediction_confidences = weed_probs[weed_mask]
        metrics['avg_weed_prediction_confidence'] = torch.mean(weed_prediction_confidences).item()
        metrics['min_weed_prediction_confidence'] = torch.min(weed_prediction_confidences).item()
        
        # How confident the model is about its weed predictions
        max_probs, _ = torch.max(prob_map, dim=0)
        weed_max_confidences = max_probs[weed_mask]
        metrics['avg_weed_max_confidence'] = torch.mean(weed_max_confidences).item()
    else:
        metrics['avg_weed_prediction_confidence'] = 0.0
        metrics['min_weed_prediction_confidence'] = 0.0
        metrics['avg_weed_max_confidence'] = 0.0
    
    return metrics

def compute_class_specific_confidence(prob_map, target_class_idx):
    """
    Compute confidence metrics for a specific class.

    Args:
        prob_map (Tensor): shape (C, H, W), softmax probabilities over classes.
        target_class_idx (int): Index of the target class.

    Returns:
        dict: Dictionary containing class-specific confidence metrics.
    """
    # Get probability map for the target class
    class_probs = prob_map[target_class_idx]  # shape (H, W)
    
    # Get predicted class for each pixel
    pred_classes = torch.argmax(prob_map, dim=0)  # shape (H, W)
    
    # Find pixels predicted as target class
    class_mask = (pred_classes == target_class_idx)
    
    metrics = {}
    
    # Overall class probability statistics
    metrics[f'avg_class_{target_class_idx}_prob'] = torch.mean(class_probs).item()
    metrics[f'max_class_{target_class_idx}_prob'] = torch.max(class_probs).item()
    
    # Class detection metrics
    total_pixels = pred_classes.numel()
    class_pixels = torch.sum(class_mask).item()
    metrics[f'class_{target_class_idx}_pixel_ratio'] = class_pixels / total_pixels
    
    # Confidence of class predictions
    if class_pixels > 0:
        class_prediction_confidences = class_probs[class_mask]
        metrics[f'avg_class_{target_class_idx}_prediction_confidence'] = torch.mean(class_prediction_confidences).item()
        
        # Overall prediction confidence for this class
        max_probs, _ = torch.max(prob_map, dim=0)
        class_max_confidences = max_probs[class_mask]
        metrics[f'avg_class_{target_class_idx}_max_confidence'] = torch.mean(class_max_confidences).item()
    else:
        metrics[f'avg_class_{target_class_idx}_prediction_confidence'] = 0.0
        metrics[f'avg_class_{target_class_idx}_max_confidence'] = 0.0
    
    return metrics

def compute_incons(diff):
    return diff.sum()/(diff.shape[0]*diff.shape[1])

def sort_images_by_entropy(model, folder_path, input_shape, num_classes=3):
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.[jp][pn]g')))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not image_files:
        print("No images found in folder.")
        return
    list_index_ent = []
    for idx, img_path in tqdm(enumerate(image_files),
                              total=len(image_files),
                              desc="Processing images for entropy"):
        img_tensor = load_image(img_path, input_shape)
        img_flip = transF.hflip(img_tensor)
        img_tensor = img_tensor.to(device)
        img_flip = img_flip.to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            prob_map = F.softmax(logits.squeeze(0), dim=0)
        entropy = compute_image_entropy(prob_map)
        confidence = compute_class_specific_confidence(prob_map, target_class_idx=1)['avg_class_1_prediction_confidence']  # Assuming class 1 is the crop class
        weed_confidence = compute_weed_confidence(prob_map, weed_class_idx=2)
        list_index_ent.append((img_path, entropy, confidence, weed_confidence['avg_weed_prediction_confidence']))
    list_index_ent = [(tensor, float(entropy), float(confidence), float(weed_conf)) for tensor, entropy, confidence, weed_conf in list_index_ent]
    list_index_ent.sort(key=lambda x: x[3], reverse=True)
    with open('entropy_sorted.txt', 'w') as f:
        for img_path, ent_value, conf_value, weed_conf_value in list_index_ent:
            f.write(f"{img_path},{ent_value},{conf_value},{weed_conf_value}\n")
    return list_index_ent

# def query_and_update():
#     sorted_idxs = sort_images_by_entropy()
#     sorted_idxs = [int(item) for item in sorted_idxs]
#     new_samples = int(len(list_train_imgs)*self.args.sampling_image_budget)
#     print ('New images:', sorted_idxs[:new_samples])
#     self.idxs_lb  = np.concatenate((self.idxs_lb, sorted_idxs[:new_samples]), axis=0)
#     self.idxs_unlb = [item for item in self.idxs_unlb if item not in self.idxs_lb]
    

def main():
    parser = argparse.ArgumentParser(description="TorchScript Segmentation Inference")
    parser.add_argument('model_path', help='Path to TorchScript model (.pt)')
    parser.add_argument('input_path', help='Path to input image, video (.mp4), or folder')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[512, 512], help='Input shape (H W)')
    parser.add_argument('--mode', choices=['image', 'video', 'folder', 'entropy'], default='image', help='Input type')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of highest/lowest entropy images to visualize')
    args = parser.parse_args()

    model = torch.jit.load(args.model_path, map_location='cuda')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # if torch.cuda.is_available():
    #     model = model.cuda()

    input_shape = (1, 3, args.input_shape[0], args.input_shape[1])

    if args.mode == 'image':
        img = cv2.imread(args.input_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = load_image(args.input_path, input_shape)
        t, _ = infer_and_time(model, img_tensor, rgb_img, num_classes=3, show=True)
        print(f"Inference time: {t:.3f}s, FPS: {1.0/t:.2f}")
    elif args.mode == 'video':
        infer_on_video(model, args.input_path, input_shape, num_classes=3)
    elif args.mode == 'folder':
        infer_on_folder(model, args.input_path, input_shape, num_classes=3)
    elif args.mode == 'entropy':
        sorted_list = sort_images_by_entropy(model, args.input_path, input_shape, num_classes=3)
        if sorted_list:
            print("Sorted images by entropy:")
            for img_path, ent_value, conf_value, weed_conf_value in sorted_list[:args.n_samples]:  # Show top n_samples
                print(f"{img_path}: {ent_value:.4f}, {conf_value:.4f}, {weed_conf_value:.4f}")
            print("...")
            for img_path, ent_value, conf_value, weed_conf_value in sorted_list[-args.n_samples:]:  # Show bottom n_samples
                print(f"{img_path}: {ent_value:.4f}, {conf_value:.4f}, {weed_conf_value:.4f}")

            # Visualize entropy results
            print(f"\nVisualizing {args.n_samples} highest and lowest entropy images...")
            visualize_entropy_results(model, sorted_list, input_shape, n=args.n_samples, num_classes=3)
        else:
            print("No images found or processed.")

if __name__ == '__main__':
    main()