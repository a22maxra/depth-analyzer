import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

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
    dataset = scipy.io.loadmat(file_path)
    images_arr = dataset['images']   # shape: (448, 608, 3, 10)
    depths_arr = dataset['depths']     # shape: (448, 608, 10)

    # Print information for the images array
    print("Images Array Information:")
    print("Shape:", images_arr.shape)
    print("Data Type:", images_arr.dtype)
    print("Min value:", images_arr.min())
    print("Max value:", images_arr.max())
    print("Length (number of images):", images_arr.shape[-1])  # Assuming last dimension is number of images

    # Print information for the depths array
    print("\nDepths Array Information:")
    print("Shape:", depths_arr.shape)
    print("Data Type:", depths_arr.dtype)
    print("Min value:", depths_arr.min())
    print("Max value:", depths_arr.max())
    print("Length (number of depth maps):", depths_arr.shape[-1])  # Assuming last dimension is number of depth maps

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
    # Extract only the valid (unmasked) values
    gt_valid = gt
    pred_valid = pred

    # Compute threshold ratios on the valid data
    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    # Compute error metrics using the valid data only
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
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
                              min_depth=0, max_depth=80.0):
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
    alt_errors_list = []

    # Convert inverse relative depth to relative depth
    def convert_inverse_depth(rel_depth):
        inv_depth = 1 / rel_depth
        return inv_depth
    
    def create_valid_mask(depth_map, min_depth, max_depth):
        return np.logical_and.reduce((
            np.isfinite(depth_map),
            depth_map > min_depth,
            depth_map < max_depth
        ))
    
    images = dataset["images"].transpose(3, 0, 1, 2)
    depths = dataset["depths"].transpose(2, 0, 1)

    for image, gt_depth in zip(images, depths):
        # Obtain model prediction
        pred_output = model(image)
        # pred_output = (pred_output - pred_output.min()) / (pred_output.max() - pred_output.min()) * 255.0
        # pred_output = pred_output.astype(np.uint8)
        
        # Use the same max_depth threshold for both gt and predictions.
        pred_mask = create_valid_mask(pred_output, min_depth, max_depth)
        gt_mask = create_valid_mask(gt_depth, min_depth, max_depth)
        combined_mask = np.logical_and(gt_mask, pred_mask)

        if not np.any(combined_mask):
            raise ValueError("No valid pixels in combined mask. Depth map is incorrect.")
        
        # Only keep valid pixels for evaluation
        pred_depth_valid = np.ma.array(pred_output, mask=~combined_mask)
        gt_depth_valid = np.ma.array(gt_depth, mask=~combined_mask)

        if do_convert:
            pred_depth_valid = convert_inverse_depth(pred_depth_valid)
            print("Normalized and inverted relative depth map")
        
        median_gt = np.median(gt_depth_valid.compressed())
        median_pred = np.median(pred_depth_valid.compressed())
        ratio = median_gt / median_pred
        pred_depth_valid = pred_depth_valid * ratio

        # Convert in other way?
        #
        #
        # Normalize the predicted inverse (or relative) depth map to [0, 1]
        copy_depth_abs = gt_depth.copy()
        copy_depth_rel = pred_output.copy()
        copy_depth_rel = np.ma.array(copy_depth_rel, mask=~combined_mask)
        copy_depth_abs = np.ma.array(copy_depth_abs, mask=~combined_mask)
        depth_rel_norm = (copy_depth_rel - np.min(copy_depth_rel)) / (np.max(copy_depth_rel) - np.min(copy_depth_rel))

        # Use the ground truth absolute depth to compute calibration parameters.
        d_min = np.min(copy_depth_abs)
        d_max = np.max(copy_depth_abs)

        # Compute transformation parameters A and B based on the ground truth range.
        A = (1.0 / d_min) - (1.0 / d_max)
        B = 1.0 / d_max

        # Convert the normalized predicted inverse depth to an absolute depth map.
        predicted_depth = 1.0 / (A * depth_rel_norm + B)

        # Now compute error metrics comparing the predicted absolute depth to the ground truth.
        alt_errors = compute_errors(copy_depth_abs, predicted_depth)
        alt_errors_list.append(alt_errors)

        #
        #
        #

        # Compute errors for the current sample
        errors = compute_errors(gt_depth_valid, pred_depth_valid)
        errors_list.append(errors)
    
    # Aggregate errors over all samples

    print("Alt Aggregated error metrics:")
    alt_aggregated_errors = {k: np.mean([e[k] for e in alt_errors_list]) for k in alt_errors_list[0].keys()}
    for key, value in alt_aggregated_errors.items():
        print(f"{key}: {value:.4f}")

    aggregated_errors = {k: np.mean([e[k] for e in errors_list]) for k in errors_list[0].keys()}
    
    return aggregated_errors