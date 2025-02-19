import scipy.io
import numpy as np

def create_mat_subset(original_mat_path, new_mat_path, n=10):
    """
    Loads the .mat file at original_mat_path and saves a new .mat file at new_mat_path
    containing only the first n images and their corresponding depths.
    
    Assumes that the original .mat file has keys 'images' and 'depths' where:
      - 'images' has shape (H, W, C, N)
      - 'depths' has shape (H, W, N) or (H, W, 1, N)
    """
    data = scipy.io.loadmat(original_mat_path)
    # Adjust these keys if needed.
    images = data['images']   # shape: (H, W, C, N)
    depths = data['depths']   # shape: (H, W, N) or maybe (H, W, 1, N)
    
    # Extract the first n images and depths.
    images_subset = images[:, :, :, :n]
    depths_subset = depths[:, :, :n] if depths.ndim == 3 else depths[:, :, 0, :n]
    
    # Save to a new .mat file.
    new_data = {'images': images_subset, 'depths': depths_subset}
    scipy.io.savemat(new_mat_path, new_data)
    print(f"Saved subset with {n} images to {new_mat_path}")

if __name__ == '__main__':
    original_mat = 'nyu_depth_v2_cropped.mat'
    new_mat = 'nyu_depth_v2_cropped_10.mat'
    create_mat_subset(original_mat, new_mat, n=10)