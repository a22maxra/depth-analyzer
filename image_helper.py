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
    print(f"\nDepth shape: {depths.shape}")       
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

def load_mat_dataset(file_path):
    import scipy.io
    data = scipy.io.loadmat(file_path)
    images_arr = data['images']   # shape: (448, 608, 3, 10)
    depths_arr = data['depths']     # shape: (448, 608, 10)
    
    dataset = []
    num_images = images_arr.shape[3]
    for i in range(num_images):
        image = images_arr[:, :, :, i]  # shape: (448, 608, 3)
        depth = depths_arr[:, :, i]       # shape: (448, 608)
        dataset.append((image, depth))
    return dataset

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
    
        True Depth = 1 / (A * depth_rel_norm + B)
    
    where:
        A = (1/d_min - 1/d_max)
        B = 1/d_max
        depth_rel_norm is the normalized inverse depth (range 0 to 1)
    
    Args:
        depth_rel_norm (np.array): Normalized inverse depth map (values between 0 and 1).
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

def convert_inverse_depth(inv_depth):
    epsilon = 1e-6 # Epsilon avoids dividing by 0
    return 1 / (inv_depth + epsilon)

def compute_errors(gt, pred):
    """
    Compute error metrics between ground truth and predicted depth.
    """
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3
    }

def evaluate_model_on_dataset(model, dataset, do_convert=True,
                              min_depth=0, max_depth=80.0, use_median_scaling=True):
    """
    Evaluate a monocular depth estimation model on a dataset that may include infinite depths.
    
    Parameters:
      model: A callable that takes an input image and returns a prediction.
      dataset: An iterable that yields (image, gt_depth) pairs, where gt_depth is a 2D numpy array.
      do_convert (bool): If True, convert the model output assuming it is an inverse depth map.
      min_depth (float): Minimum valid depth value.
      max_depth (float): Maximum valid depth value.
      use_median_scaling (bool): Whether to perform median scaling for each sample.
    
    Returns:
      dict: Aggregated error metrics across the dataset.
    """
    errors_list = []
    scaling_ratios = []
    
    # Convert inverse relative depth to relative depth
    def convert_inverse_depth(inv_depth):
        epsilon = 1e-6  # to avoid division by zero
        return 1.0 / (inv_depth + epsilon)
    
    def create_valid_mask(depth_map, min_depth, max_depth):
        return np.logical_and.reduce((
            np.isfinite(depth_map),
            depth_map > min_depth,
            depth_map < max_depth
        ))
    
    for image, gt_depth in dataset:
        # Obtain model prediction
        pred_output = model(image)
        # Create a valid mask: use only pixels that are finite and in range.
        pred_mask = create_valid_mask(pred_output, min_depth, max_depth = True)

        if not np.any(pred_mask):
            raise ValueError("No valid pixels in prediction mask. Model output may be invalid.")
    
        # If do_convert is True, convert inverse depth to depth
        if do_convert:
            pred_depth = convert_inverse_depth(pred_output)
        else:
            pred_depth = pred_output
        
        # Create a valid mask: use only ground truth pixels that are finite and in range.
        valid_mask = create_valid_mask(gt_depth, min_depth, max_depth)
    
        if not np.any(valid_mask):
            raise ValueError("No valid pixels in ground truth mask.")
        
        combined_mask = np.logical_and(valid_mask, pred_mask)
        pred_depth_valid = np.ma.array(pred_depth, mask=~combined_mask)
        gt_depth_valid = np.ma.array(gt_depth, mask=~combined_mask)

        median_gt = np.median(gt_depth_valid)
        median_pred = np.median(pred_depth_valid)
        ratio = median_gt / median_pred
        scaling_ratios.append(ratio)
        pred_depth_valid = pred_depth_valid * ratio
        
        # Compute errors for the current sample
        errors = compute_errors(gt_depth_valid, pred_depth_valid)
        errors_list.append(errors)
    
    # Aggregate errors over all samples
    aggregated_errors = {k: np.mean([e[k] for e in errors_list]) for k in errors_list[0].keys()}
    
    return aggregated_errors