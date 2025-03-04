import os
from PIL import Image
import numpy as np

# Load the scannet dataset
# Path to the downloaded txt file
# From https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/data_split/scannet/scannet_val_sampled_list_800_1.txt
txt_file = './datasets/scannet/scannet_val_sampled_list_800_1.txt'

# Open and read the file
with open(txt_file, 'r') as f:
    lines = f.readlines()

# Parse the lines: each line has: image_path, gt_depth_path
data_list = []
for line in lines:
    parts = line.strip().split()
    if len(parts) < 2:
        continue  # Skip malformed lines
    img_path, depth_path = parts[0], parts[1]
    data_list.append((img_path, depth_path))

base_dir = './datasets/scannet/'


def load_scannet(max_images=None):
    depths_list = []
    images_list = []
    
    for img_path, depth_path in data_list:
        if max_images != None and max_images <= len(images_list): break
        if depth_path != "None" and img_path != "None":
            img_full_path = os.path.join(base_dir, img_path)
            depth_full_path = os.path.join(base_dir, depth_path)

            depth = Image.open(depth_full_path)
            depth_np = np.array(depth)
            depth_decoded = depth_np / 1000.0  # Convert to meters

            image = Image.open(img_full_path).convert('RGB')
            image_np = np.array(image)

            depths_list.append(depth_decoded)
            images_list.append(image_np)

            print(f"\rLoading scannet. Current image count: {len(images_list)}/{len(data_list)}", end="")


    depths = np.stack(depths_list, axis=0)
    images = np.stack(images_list, axis=0)

    print("\nShape of depths array:", depths.shape)
    print("Shape of images array:", images.shape)

    zipped_dataset = {'images': images, 'depths': depths}
    return zipped_dataset