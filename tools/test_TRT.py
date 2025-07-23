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
    np.copyto(inputs[0]['host'], input_image.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

def preprocess(img_path, input_shape):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[2], input_shape[3]))
    img = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, :]

def postprocess(output, input_shape):
    output = output.reshape(1, 3, 128, 128)
    seg_map = np.argmax(output, axis=1).astype(np.uint8)
    seg_map = cv2.resize(seg_map[0], (input_shape[2], input_shape[3]), interpolation=cv2.INTER_NEAREST)
    # seg_map = np.expand_dims(seg_map, axis=-1)  # Add channel dimension
    return seg_map

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

    def run_inference(img):
        img_tensor = preprocess(img, input_shape)
        t0 = time.time()
        output = infer(engine, context, inputs, outputs, bindings, stream, img_tensor)
        t1 = time.time()
        seg_map = postprocess(output, input_shape)
        return t1 - t0, seg_map

    if args.mode == 'image':
        t, seg_map = run_inference(args.input_path)
        print(f"Inference time: {t:.3f}s, FPS: {1.0/t:.2f}")
        # Visualization code can be added here
    elif args.mode == 'folder':
        image_files = sorted(glob.glob(os.path.join(args.input_path, '*.[jp][pn]g')))
        frame_times = []
        for idx, img_path in enumerate(image_files):
            t, seg_map = run_inference(img_path)
            frame_times.append(t)
            print(f"Image {idx+1}/{len(image_files)}: {1.0/t:.2f} FPS")
        if frame_times:
            print(f"Average FPS: {1.0/np.mean(frame_times):.2f}")
    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.input_path)
        frame_times = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_idx = random.randint(0, total_frames - 1)
        saved = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print('input shape:', img.shape)
            img_resized = cv2.resize(img, (input_shape[2], input_shape[3]))
            img_float = img_resized.astype(np.float32)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            img_norm = (img_float - mean) / std
            img_transposed = img_norm.transpose(2, 0, 1)
            img_tensor = img_transposed[np.newaxis, :]
            t0 = time.time()
            output = infer(engine, context, inputs, outputs, bindings, stream, img_tensor)
            t1 = time.time()
            seg_map = postprocess(output, input_shape)
            frame_times.append(t1 - t0)
            print(f"Frame {frame_count+1}: {1.0/(t1-t0):.2f} FPS")
            # Save random frame and its segmentation map
            if frame_count == random_frame_idx and not saved:
                # Save RGB input
                cv2.imwrite("random_frame_rgb.png", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                # Save colored segmentation map
                seg_color = colorize_seg_map(seg_map)
                cv2.imwrite("random_frame_seg.png", cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
                saved = True
            frame_count += 1
        cap.release()
        if frame_times:
            print(f"Average FPS: {1.0/np.mean(frame_times):.2f}")

if __name__ == '__main__':
    main()
