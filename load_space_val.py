# Load the "In_Space_Type" validation dataset

import os
from PIL import Image
import numpy as np

def load_pfm(file_path):
    """
    Load a PFM file into a numpy array.
    PFM is a simple format for storing floating point images.
    """
    with open(file_path, 'rb') as f:
        header = f.readline().rstrip().decode('utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception("Not a PFM file.")
        
        dims_line = f.readline().decode('utf-8').strip()
        width, height = map(int, dims_line.split())
        
        scale = float(f.readline().rstrip().decode('utf-8'))
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        
        # Reshape data to the appropriate format
        if color:
            shape = (height, width, 3)
        else:
            shape = (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in a bottom-up order
        
        return data

def load_in_space_type(max_images=None):
    base_dir = './datasets/in_space_type/'
    images_list = []
    depths_list = []
        
    # List all files in the dataset directory
    files = os.listdir(base_dir)
    # Filter to get only the image files matching the naming convention *_L.jpg
    image_files = sorted([f for f in files if f.endswith('_L.jpg')])
    
    for image_file in image_files:
        if max_images != None and max_images <= len(images_list): break

        # Full path to the image file
        img_full_path = os.path.join(base_dir, image_file)
        # Construct corresponding depth file name:
        # e.g. "0084_L.jpg" -> "0084.pfm"
        base_name = image_file.replace('_L.jpg', '')
        depth_file = base_name + '.pfm'
        depth_full_path = os.path.join(base_dir, depth_file)
        
        if not os.path.exists(depth_full_path):
            print(f"Depth file {depth_full_path} not found, skipping {image_file}.")
            continue
        
        # Load image and convert to RGB
        image = Image.open(img_full_path).convert('RGB')
        image_np = np.array(image)
        
        # Load the depth map from the pfm file
        depth_np = load_pfm(depth_full_path)
        
        images_list.append(image_np)
        depths_list.append(depth_np)
        
        print(f"\rLoading in_space_type. Current image count: {len(images_list)}/{len(image_files)}", end="")
    
    images = np.stack(images_list, axis=0)
    depths = np.stack(depths_list, axis=0)
    
    print("\nShape of images array:", images.shape)
    print("Shape of depths array:", depths.shape)
    
    dataset = {'images': images, 'depths': depths}
    return dataset
