import numpy as np
from PIL import Image
import os


def load_wuav(max_images=None):
    depths_list = []
    images_list = []

    depth_folder = './datasets/wuav/depth'
    img_folder = './datasets/wuav/img'

    for depth_file in os.listdir(depth_folder):
        if max_images != None and max_images <= len(images_list): break
        if depth_file.endswith('.npy'):
            base_name = os.path.splitext(depth_file)[0]

            img_file = base_name + '.png'
            depth_path = os.path.join(depth_folder, depth_file)
            img_path = os.path.join(img_folder, img_file)

            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            depth_np = np.load(depth_path)
            depth_np = np.squeeze(depth_np)

            images_list.append(image_np)
            depths_list.append(depth_np)

            print(f"\rFound {len(images_list)} image-depth pairs", end="")
    
    images = np.stack(images_list, axis=0)
    depths = np.stack(depths_list, axis=0)

    print("Shape of images array:", images.shape)
    print("Shape of depths array:", depths.shape)

    zipped_dataset = {'images': images, 'depths': depths}
    return zipped_dataset
