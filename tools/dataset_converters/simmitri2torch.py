from os.path import join
import datumaro as dm
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Use TkAgg backend for interactive viewing
matplotlib.use('TkAgg')

def reindex_classes():
    splits = ['train', 'val']
    for split in splits:
        files = glob(f'/mnt/e/projects/TFP/2025-09-02_Testdatensatz/cityscapes/gtFine/{split}/*_labelIds.png')
        for file in files:
            img = Image.open(file)
            img = np.array(img).astype(np.uint8)
            plt.imshow(img)
            plt.show(block=True)  # or just plt.show()
            # plt.pause(0.001) 
            img = img - 1
            # plt.imshow(img)
            # plt.show()
            img = Image.fromarray(img)
            img.save(file)

def process_simmitri_dataset(dataset_dir:str):
    """
    
    """
    dataset_format = dm.Dataset.detect(dataset_dir)
    dataset = dm.Dataset.import_from(dataset_dir, dataset_format)
    print('Creating Train and Validation splits')
    splits = (('train', 0.9), ('val', 0.1))
    # 1. Split dataset into train and validation split
    dataset = dataset.transform(method='random_split',
                                splits=splits, seed=42)
    subsets = list(dataset.subsets().keys())
    print("Subset candidates:", subsets)
    # 2. Save the dataset as panoptic segmentation.
    # NOTE! At the moment, there is no way to tell apart the stuff and thing classes. 
    # HACK! Manually edit the isthing attribute for each class in the output json file. 
    # Convert dataset into Panoptic with Stuff -> Background and Onions &&& Things -> all Weeds
    print('Exporting dataset into coco_panoptic format under folder: \n {}'.format(dataset))
    dataset.export(format='coco_panoptic', save_dir=join(dataset_dir, 'coco_panoptic'))
    
    # 3. Convert dataset into Cityscapes format in the original segmentation format. 
    print('Exporting dataset into cityscapes format under folder: \n {}'.format(dataset))
    dataset.export(format='cityscapes', save_dir=join(dataset_dir, 'cityscapes'), save_media=True)
    # LabelIds have to be zero indexed. 
    reindex_classes()
    # Generating instance based dataset for onions and weeds
    dataset.transform(method='remap_labels',
                    mapping={'Ground': '',
                            'Onions': 'Onions',
                            'Gaensefuss': 'Gaensefuss',
                            'Hirse': 'Hirse',
                            'Labkraut': 'Labkraut'},
                    default='keep')
    
    dataset.export(format='coco_instances', save_dir=join(dataset_dir, 'coco_detection_with_crops_and_weeds'))
    dataset.export(format='yolo_ultralytics', save_dir=join(dataset_dir, 'yolo_ultralytics_with_crops_and_weeds'),
                   save_media=True)
    # Generating instance based dataset without onions and only weeds.
    dataset.transform(method='remap_labels',
                                        mapping={'Ground': '',
                                                 'Onions': '',
                                                 'Gaensefuss': 'Gaensefuss',
                                                 'Hirse': 'Hirse',
                                                 'Labkraut': 'Labkraut'},
                                        default='keep')
    dataset.export(format='coco_instances', save_dir=join(dataset_dir, 'coco_detection_no_crops_only_weeds'))
    dataset.export(format='yolo_ultralytics', save_dir=join(dataset_dir, 'yolo_ultralytics_no_crops_only_weeds'),
                   save_media=True)

    # [Optional]TODO! Tile dataset into lower resolution images


if __name__ == "__main__":
    dataset_dir = '/mnt/e/projects/TFP/2025-09-02_Testdatensatz'
    process_simmitri_dataset(dataset_dir=dataset_dir)