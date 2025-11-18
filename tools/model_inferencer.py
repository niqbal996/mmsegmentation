from mmseg.apis import init_model, inference_model, show_result_pyplot
from glob import glob
import os
from tqdm import tqdm
from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
config_path = 'mmseg_forked/configs/mask2former/mask2former_r50_simmetry-1024x1024.py'
# checkpoint_path = '/netscratch/naeem/tfp_project/mask2former_r50_simmetry_weeds_3_classes/best_mIoU_iter_5000.pth'
checkpoint_path = '/mnt/e/projects/TFP/models/mask2former_r50_p4ai_3_classes_baseline/best_mIoU_iter_2000.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')
# img_path = '/netscratch/naeem/tfp_project/simmetry_dataset_mini/yolo_ultralytics_weeds_3_classes/images/val'
img_path = '/mnt/e/projects/TFP/AMZ-POC_COCO_71_27-10-2025/cityscapes/cityscapes/leftImg8bit/val'
# for camera in cameras:
#     img_path = f'/netscratch/naeem/tfp_project/test_dataset/{camera}'
# out_dir = f'/netscratch/naeem/tfp_project/test_dataset/mask2former_r50_weeds_3_classes'
out_dir = f'/mnt/e/projects/TFP/inference_results/p4ai_real_data/predictions_only'
os.makedirs(out_dir, exist_ok=True)
img_paths = glob(f'{img_path}/*.png')
model.dataset_meta['palette'][1] = [255, 0, 0]
model.dataset_meta['palette'][2] = [0, 255, 0]
for img in tqdm(img_paths, total=len(img_paths)):
# inference on given image
result = inference_model(model, img_paths)

# display the segmentation result
vis_image = show_result_pyplot(model, img_path, result)

# save the visualization result, the output image would be found at the path `work_dirs/result.png`
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# Modify the time of displaying images, note that 0 is the special value that means "forever"
vis_image = show_result_pyplot(model, img_path, result, wait_time=5)