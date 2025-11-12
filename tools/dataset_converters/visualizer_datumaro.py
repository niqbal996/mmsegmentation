from os.path import join
import datumaro as dm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Qt4Agg', etc.

dataset_dir = '/mnt/e/projects/TFP/2025-09-02_Testdatensatz'
save_dir = '/mnt/e/projects/TFP/simmetry_converted_dataset/weed_gt_visualization'
dataset_format = dm.Dataset.detect(dataset_dir)
dataset = dm.Dataset.import_from(dataset_dir, dataset_format)
subset = list(dataset.subsets().keys())[0] 
def get_ids(dataset: dm.Dataset, subset: str):
    ids = []
    for item in dataset:
        if item.subset == subset:
            ids += [item.id]

    return ids

ids = get_ids(dataset, subset)

visualizer = dm.Visualizer(dataset, 
                           figsize=(16, 12), 
                           alpha=0.7,
                           color_cycles=['red', 'teal', 'yellow'],
                           bbox_linewidth=2,)

# Set up interactive navigation
current_idx = 0
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

def on_key(event):
    global current_idx
    if event.key == 'right' or event.key == 'n':
        current_idx = (current_idx + 1) % len(ids)
        update_display()
    elif event.key == 'left' or event.key == 'p':
        current_idx = (current_idx - 1) % len(ids)
        update_display()
    elif event.key == 'q' or event.key == 'escape':
        plt.close()

def update_display():
    ax.clear()
    print(f"Image ID: {ids[current_idx]} ({current_idx + 1}/{len(ids)})")
    visualizer.vis_one_sample(ids[current_idx], subset, ax=ax)
    ax.set_title(f"Image: {ids[current_idx]} ({current_idx + 1}/{len(ids)}) - Use arrow keys/n/p to navigate, q to quit")
    plt.draw()

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial display
update_display()
plt.tight_layout()
print("Use arrow keys or 'n'/'p' to navigate, 'q' to quit")
plt.show()
plt.savefig(join(save_dir, f'{ids[current_idx]}.png'))
