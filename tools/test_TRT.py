import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import argparse
import os
import glob
import time
import random

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = int(np.prod(shape))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def infer(engine, context, inputs, outputs, bindings, stream, input_image):
    """Run inference with timing."""
    t_start = time.perf_counter()
    
    np.copyto(inputs[0]['host'], input_image.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    
    t_end = time.perf_counter()
    inference_time = (t_end - t_start) * 1000  # ms
    
    return outputs[0]['host'], inference_time

def preprocess(img_path, input_shape):
    """Preprocess image with timing."""
    t_start = time.perf_counter()
    
    # Load image
    t_load_start = time.perf_counter()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t_load_end = time.perf_counter()
    
    # Resize and normalize
    img = cv2.resize(img, (input_shape[2], input_shape[3]))
    img = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    
    t_end = time.perf_counter()
    
    timing = {
        'load_time': (t_load_end - t_load_start) * 1000,  # ms
        'preprocess_time': (t_end - t_start) * 1000  # ms (total including load)
    }
    
    return img[np.newaxis, :], timing

def postprocess(output, input_shape):
    """Postprocess output with timing."""
    t_start = time.perf_counter()
    
    output = output.reshape(1, 3, 128, 128)
    seg_map = np.argmax(output, axis=1).astype(np.uint8)
    seg_map = cv2.resize(seg_map[0], (input_shape[2], input_shape[3]), interpolation=cv2.INTER_NEAREST)
    
    t_end = time.perf_counter()
    postprocess_time = (t_end - t_start) * 1000  # ms
    
    return seg_map, postprocess_time

def colorize_seg_map(seg_map):
    # Assign class 0: red, class 1: green, class 2: black
    color_map = np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 0],      # Black
    ], dtype=np.uint8)
    return color_map[seg_map]

def main():
    parser = argparse.ArgumentParser(description="TensorRT Segmentation Inference")
    parser.add_argument('engine_path', help='Path to TensorRT engine file')
    parser.add_argument('input_path', help='Path to input image, video (.mp4), or folder')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[1024, 1024], help='Input shape (H W)')
    parser.add_argument('--mode', choices=['image', 'video', 'folder'], default='image', help='Input type')
    args = parser.parse_args()

    input_shape = (1, 3, args.input_shape[0], args.input_shape[1])
    engine = load_engine(args.engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    def run_inference(img_path):
        # Preprocess with timing
        img_tensor, preprocess_timing = preprocess(img_path, input_shape)
        
        # Inference with timing
        output, inference_time = infer(engine, context, inputs, outputs, bindings, stream, img_tensor)
        
        # Postprocess with timing
        seg_map, postprocess_time = postprocess(output, input_shape)
        
        # Calculate total time
        total_time = preprocess_timing['preprocess_time'] + inference_time + postprocess_time
        
        timing_info = {
            'load_time': preprocess_timing['load_time'],
            'preprocess_time': preprocess_timing['preprocess_time'],
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'total_time': total_time
        }
        
        return seg_map, timing_info

    if args.mode == 'image':
        seg_map, timing = run_inference(args.input_path)
        print(f"=== Timing Breakdown ===")
        print(f"Image Load:    {timing['load_time']:.2f} ms")
        print(f"Preprocessing: {timing['preprocess_time']:.2f} ms")
        print(f"Inference:     {timing['inference_time']:.2f} ms")
        print(f"Postprocess:   {timing['postprocess_time']:.2f} ms")
        print(f"Total Time:    {timing['total_time']:.2f} ms")
        print(f"FPS:           {1000.0/timing['total_time']:.2f}")
        # Visualization code can be added here
    elif args.mode == 'folder':
        image_files = sorted(glob.glob(os.path.join(args.input_path, '*.[jp][pn]g')))
        all_timings = {
            'load_times': [],
            'preprocess_times': [],
            'inference_times': [],
            'postprocess_times': [],
            'total_times': []
        }
        
        print(f"Processing {len(image_files)} images...")
        print("=" * 80)
        
        for idx, img_path in enumerate(image_files):
            seg_map, timing = run_inference(img_path)
            
            # Store timings
            all_timings['load_times'].append(timing['load_time'])
            all_timings['preprocess_times'].append(timing['preprocess_time'])
            all_timings['inference_times'].append(timing['inference_time'])
            all_timings['postprocess_times'].append(timing['postprocess_time'])
            all_timings['total_times'].append(timing['total_time'])
            
            fps = 1000.0 / timing['total_time']
            print(f"Image {idx+1:3d}/{len(image_files)}: "
                  f"Load={timing['load_time']:5.1f}ms | "
                  f"Pre={timing['preprocess_time']:5.1f}ms | "
                  f"Inf={timing['inference_time']:5.1f}ms | "
                  f"Post={timing['postprocess_time']:4.1f}ms | "
                  f"Total={timing['total_time']:6.1f}ms | "
                  f"FPS={fps:5.1f}")
        
        if all_timings['total_times']:
            print("=" * 80)
            print("=== Average Timing Summary ===")
            print(f"Average Load Time:       {np.mean(all_timings['load_times']):.2f} ms")
            print(f"Average Preprocessing:   {np.mean(all_timings['preprocess_times']):.2f} ms")
            print(f"Average Inference:       {np.mean(all_timings['inference_times']):.2f} ms")
            print(f"Average Postprocessing:  {np.mean(all_timings['postprocess_times']):.2f} ms")
            print(f"Average Total Time:      {np.mean(all_timings['total_times']):.2f} ms")
            print(f"Average FPS:             {1000.0/np.mean(all_timings['total_times']):.2f}")
            print("=" * 80)
    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.input_path)
        all_timings = {
            'preprocess_times': [],
            'inference_times': [],
            'postprocess_times': [],
            'total_times': []
        }
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_idx = random.randint(0, total_frames - 1)
        saved = False
        
        print(f"Processing video with {total_frames} frames...")
        print("=" * 80)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            t_pre_start = time.perf_counter()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (input_shape[2], input_shape[3]))
            img_float = img_resized.astype(np.float32)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            img_norm = (img_float - mean) / std
            img_transposed = img_norm.transpose(2, 0, 1)
            img_tensor = img_transposed[np.newaxis, :]
            t_pre_end = time.perf_counter()
            preprocess_time = (t_pre_end - t_pre_start) * 1000
            
            # Inference
            output, inference_time = infer(engine, context, inputs, outputs, bindings, stream, img_tensor)
            
            # Postprocess
            seg_map, postprocess_time = postprocess(output, input_shape)
            
            total_time = preprocess_time + inference_time + postprocess_time
            
            # Store timings
            all_timings['preprocess_times'].append(preprocess_time)
            all_timings['inference_times'].append(inference_time)
            all_timings['postprocess_times'].append(postprocess_time)
            all_timings['total_times'].append(total_time)
            
            fps = 1000.0 / total_time
            print(f"Frame {frame_count+1:4d}: "
                  f"Pre={preprocess_time:5.1f}ms | "
                  f"Inf={inference_time:5.1f}ms | "
                  f"Post={postprocess_time:4.1f}ms | "
                  f"Total={total_time:6.1f}ms | "
                  f"FPS={fps:5.1f}")
            
            # Save random frame and its segmentation map
            if frame_count == random_frame_idx and not saved:
                # Save RGB input
                cv2.imwrite("random_frame_rgb.png", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                # Save colored segmentation map
                seg_color = colorize_seg_map(seg_map)
                cv2.imwrite("random_frame_seg.png", cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
                print(f"Saved random frame {frame_count+1} visualization")
                saved = True
            frame_count += 1
            
        cap.release()
        
        if all_timings['total_times']:
            print("=" * 80)
            print("=== Video Processing Summary ===")
            print(f"Total Frames:            {len(all_timings['total_times'])}")
            print(f"Average Preprocessing:   {np.mean(all_timings['preprocess_times']):.2f} ms")
            print(f"Average Inference:       {np.mean(all_timings['inference_times']):.2f} ms")
            print(f"Average Postprocessing:  {np.mean(all_timings['postprocess_times']):.2f} ms")
            print(f"Average Total Time:      {np.mean(all_timings['total_times']):.2f} ms")
            print(f"Average FPS:             {1000.0/np.mean(all_timings['total_times']):.2f}")
            print("=" * 80)

if __name__ == '__main__':
    main()
