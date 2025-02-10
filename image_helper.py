import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Use an interactive backend (make sure your X server and tkinter are set up in WSL)
matplotlib.use('TkAgg')

def load_mat_file_images(file_path):
    data = loadmat(file_path)
    images = data['images']
    print(f"\nImages shape: {images.shape}")       
    return images

def load_mat_file_depths(file_path):
    data = loadmat(file_path)
    depths = data['depths']
    print(f"\nImages shape: {depths.shape}")       
    return depths

def inspect_mat_file(file_path):
    data = loadmat(file_path)
    print(f"\n Keys in the file: {list(data.keys())} ")

def extract_image(images_arr, index):
    image = images_arr[:, :, :, index]
    print("\nExtracted image shape:", image.shape)
    return image

def save_image(image, name="output_image.png"):
    print("\nSaving image as ", name)
    plt.imsave(name, image)

def extract_depth(depths_arr, index):
    depth = depths_arr[:, :, index]
    print("\nExtracted depth shape:", depth.shape)
    print(f"Depth Value range:, {np.min(depth), np.max(depth)}")
    return depth

def save_depth(depth, name="output_depth.png"):
    print("\nSaving depth map as ", name)
    plt.imsave(name, depth, cmap='Spectral')

def print_reduced_depth_map(depth, block_size=100):
    # Get the original dimensions
    depth = depth.astype(np.uint8)
    H, W = depth.shape

    # Crop the depth map so that H and W are divisible by block_size
    H_crop = H - (H % block_size)
    W_crop = W - (W % block_size)
    depth_cropped = depth[:H_crop, :W_crop]

    # Reshape to a 4D array where each block is block_size x block_size,
    # then average over the block dimensions (axis 1 and 3)
    reduced_depth = depth_cropped.reshape(H_crop // block_size, block_size, W_crop // block_size, block_size).mean(axis=(1, 3))
    for row in reduced_depth:
        for value in row:
            # value rounded down to integer
            print(f"{value:.1f}", end=" ")
        print()  # Newline after each row

def inverse_rel_depth_to_true_depth(depth_rel, depth_abs):
    """
    Convert a normalized inverse depth map to true depth using the formula:
    
        True Depth = 1 / (A * V_norm + B)
    
    where:
        A = (1/d_min - 1/d_max)
        B = 1/d_max
        V_norm is the normalized inverse depth (range 0 to 1)
    
    Args:
        depth_norm (np.array): Normalized inverse depth map (values between 0 and 1).
        d_min (float): Known minimum true depth (closest point in meters).
        d_max (float): Known maximum true depth (farthest point in meters).
    
    Returns:
        np.array: True depth map (in meters).
    """
    depth_rel_norm = (depth_rel - np.min(depth_rel)) / (np.max(depth_rel) - np.min(depth_rel))
    d_min = np.min(depth_abs)
    d_max = np.max(depth_abs)

    # Calculate the transformation parameters A and B
    A = (1.0 / d_min) - (1.0 / d_max)
    B = 1.0 / d_max
    
    # Compute true depth using the inversion formula
    true_depth = 1.0 / (A * depth_rel_norm + B)
    return true_depth

def compute_absolute_error(depth_pred, depth_gt):
    """
    Compute the mean absolute error between predicted and ground truth depth maps.

    Args:
        depth_pred (np.array): Predicted absolute depth map.
        depth_gt (np.array): Ground truth absolute depth map.
    
    Returns:
        float: The mean absolute error.
    """
    # Ensure the arrays have the same shape
    if depth_pred.shape != depth_gt.shape:
        raise ValueError("The predicted and ground truth depth maps must have the same shape.")
    
    # Calculate the absolute error per element
    abs_error = np.abs(depth_pred - depth_gt)
    
    # Return the mean absolute error (MAE)
    mae = np.mean(abs_error)
    return mae
