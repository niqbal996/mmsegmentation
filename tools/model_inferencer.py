from mmseg.apis import init_model, inference_model, show_result_pyplot
from glob import glob

config_path = 'mmseg_forked/configs/mask2former/mask2former_r50_simmetry-512x512.py'
checkpoint_path = '/mnt/e/projects/TFP/2025-09-02_Testdatensatz/cityscapes/best_mIoU_iter_4000.pth'
img_path = '/mnt/e/projects/TFP/2025-09-02_Testdatensatz/cityscapes/leftImg8bit/val'
img_paths = glob(f'{img_path}/*.png')

# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')

# inference on given image
result = inference_model(model, img_paths)

# display the segmentation result
vis_image = show_result_pyplot(model, img_path, result)

# save the visualization result, the output image would be found at the path `work_dirs/result.png`
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# Modify the time of displaying images, note that 0 is the special value that means "forever"
vis_image = show_result_pyplot(model, img_path, result, wait_time=5)