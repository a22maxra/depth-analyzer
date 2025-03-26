import os
from PIL import Image
import numpy as np

# Load the KITTI Eigen split dataset
# Path to the downloaded txt file
# From https://github.com/prs-eth/Marigold/blob/main/data_split/kitti/eigen_test_files_with_gt.txt
txt_file = './datasets/kitti_eigen/eigen_test_files_with_gt.txt'

# Open and read the file
with open(txt_file, 'r') as f:
    lines = f.readlines()

# Parse the lines: each line has: image_path, gt_depth_path, calibration_value
data_list = []
for line in lines:
    parts = line.strip().split()
    if len(parts) < 3:
        continue  # Skip malformed lines
    img_path, depth_path, calib = parts[0], parts[1], float(parts[2])
    data_list.append((img_path, depth_path, calib))

base_dir = './datasets/kitti_eigen/'

# Crop images/Depths to KITTI benchmark size
def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size.
    Args:
        input_img (numpy array): Input image (H x W) or (H x W x C).
    Returns:
        Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    # Get height and width from the first two dimensions.
    height, width = input_img.shape[:2]
    top_margin = height - KB_CROP_HEIGHT
    left_margin = (width - KB_CROP_WIDTH) // 2

    if input_img.ndim == 2:
        input_img = input_img[top_margin: top_margin + KB_CROP_HEIGHT,
                         left_margin: left_margin + KB_CROP_WIDTH]
        input_img = input_img.astype(np.float32) / 256.0 # Convert to depth values
        return input_img
    elif input_img.ndim == 3:
        return input_img[top_margin: top_margin + KB_CROP_HEIGHT,
                         left_margin: left_margin + KB_CROP_WIDTH, :]

def load_kitti(max_images=None):
    depths_list = []
    images_list = []
    
    for img_path, depth_path, calib in data_list:
        if max_images != None and max_images <= len(images_list): break
        if depth_path != "None" and img_path != "None":
            img_full_path = os.path.join(base_dir, img_path)
            depth_full_path = os.path.join(base_dir, depth_path)

            depth = Image.open(depth_full_path)
            depth_np = np.array(depth)

            image = Image.open(img_full_path).convert('RGB')
            image_np = np.array(image)

            cropped_depth = kitti_benchmark_crop(depth_np)
            cropped_image = kitti_benchmark_crop(image_np)

            depths_list.append(cropped_depth)
            images_list.append(cropped_image)

            print(f"\rLoading Kitti. Current image count: {len(images_list)}/{len(data_list)}", end="")

    depths = np.stack(depths_list, axis=0)
    images = np.stack(images_list, axis=0)

    print("\nShape of depths array:", depths.shape)
    print("Shape of images array:", images.shape)

    zipped_dataset = {'images': images, 'depths': depths}
    return zipped_dataset