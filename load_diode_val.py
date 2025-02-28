import os
from PIL import Image
import numpy as np

def load_diode(base_dir='./datasets/diode/val'):
    """
    Recursively loads image-depth pairs from the Diode dataset.
    
    The directory structure is assumed to be like:
      datasets/diode/val/indoors/scene_00020/scan_00186/
         00020_00186_indoors_110_000.png
         00020_00186_indoors_110_000_depth.npy
         00020_00186_indoors_110_000_depth_mask.npy

    For every PNG file, we look for a corresponding file with the same base name
    plus the suffix '_depth.npy'. The image is loaded with PIL and converted to an RGB array,
    and the depth map is loaded with np.load.
    
    Returns:
        dict: A dictionary with keys 'images' and 'depths', each a NumPy array.
    """
    images_list = []
    depths_list = []
    
    # Walk through the directory tree under base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):
                # Full path to the image
                img_full_path = os.path.join(root, file)
                # Remove the '.png' extension to get the base name
                base_name = file[:-4]
                # Construct expected depth filename
                depth_filename = base_name + '_depth.npy'
                depth_maskname = base_name + '_depth_mask.npy'
                # Check if the matching depth file exists in the same directory
                if depth_filename in files:
                    depth_full_path = os.path.join(root, depth_filename)
                    depth_mask_full_path = os.path.join(root, depth_maskname)
                    
                    # Load image and convert to RGB numpy array
                    image = Image.open(img_full_path).convert('RGB')
                    image_np = np.array(image)
                    
                    # Load depth map from .npy file
                    depth_np = np.load(depth_full_path)
                    
                    images_list.append(image_np)
                    depths_list.append(depth_np)
                    
                    print(f"\rFound {len(images_list)} image-depth pairs...", end="")
        
    # Stack lists into NumPy arrays. Make sure all images and depths share the same shape.
    images = np.stack(images_list, axis=0)
    depths = np.stack(depths_list, axis=0)
    depths = np.squeeze(depths) # Remove the last dimension
    
    print("Shape of images array:", images.shape)
    print("Shape of depths array:", depths.shape)
    
    return {'images': images, 'depths': depths}