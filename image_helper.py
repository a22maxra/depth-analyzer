import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat


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
    plt.imsave(name, depth, cmap='Spectral_r')

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

def invert_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    inverse_depth = 1.0 / depth.clip(min=eps)
    return inverse_depth

def find_scale_shift(gt, pred):
    """Compute optimal scale (s) and shift (b) using least squares."""
    A = np.vstack([pred, np.ones_like(pred)]).T
    s, b = np.linalg.lstsq(A, gt, rcond=None)[0]
    return s, b

def apply_scale_shift(pred, s, b):
    """Apply scale and shift to predicted depth map."""
    return s * pred + b

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

def rel_depth_to_true_depth(depth_rel, depth_abs):
    s, b = find_scale_shift(depth_abs, depth_rel)
    true_depth = apply_scale_shift(depth_rel, s, b)
    return true_depth


def compute_errors(gt_valid, pred_valid):

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

def evaluate_model_on_dataset(model, dataset, min_depth=0, max_depth=80.0, max_images=None, save_output=False, inverse=True):
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

    def create_valid_mask(depth_map, min_depth, max_depth):
        return np.logical_and.reduce((
            np.isfinite(depth_map),
            depth_map > min_depth,
            depth_map < max_depth
        ))
    
    images = dataset["images"].transpose(3, 0, 1, 2) # Should be (N, H, W, C)
    depths = dataset["depths"].transpose(2, 0, 1)

    if max_images is not None:
        if max_images > len(images) or max_images <= 0:
            return {}
        indices = np.random.choice(len(images), max_images, replace=False)
        images = images[indices]
        depths = depths[indices]

    for image, gt_depth in zip(images, depths):
        # Obtain model prediction
        pred_output = model(image)
        
        gt_mask = create_valid_mask(gt_depth, min_depth, max_depth)

        if not np.any(gt_mask):
            raise ValueError("No valid pixels in combined mask. Depth map is incorrect.")
        
        # Only keep valid pixels for evaluation
        pred_depth_valid = pred_output[gt_mask]
        gt_depth_valid = gt_depth[gt_mask]

        if inverse == True:
            print(" Inverse was true")
            pred_depth_valid = inverse_rel_depth_to_true_depth(pred_depth_valid, gt_depth_valid)
        else:
            pred_depth_valid = rel_depth_to_true_depth(pred_depth_valid, gt_depth_valid)

        # Compute errors for the current sample
        errors = compute_errors(gt_depth_valid, pred_depth_valid)
        errors_list.append(errors)

        if save_output and len(errors_list) <= save_output:
            save_depth(pred_output, name=f"./output/output_depth{len(errors_list)}.png")
            save_image(image, name=f"./output/output_image{len(errors_list)}.png")

        # Print current progress (images so far)
        print(f"\rProcessed {len(errors_list)} images", end="")
    aggregated_errors = {k: np.mean([e[k] for e in errors_list]) for k in errors_list[0].keys()}
    
    return aggregated_errors