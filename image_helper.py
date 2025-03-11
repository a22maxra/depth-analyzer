import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import torch

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

def save_depth(depth, name="output_depth.png", inverse=False, max_depth=80, min_depth=0.001):
    print("\nSaving depth map as ", name)
    if inverse == True: plt.imsave(name, depth, cmap='Spectral_r', vmin=min_depth, vmax=max_depth)
    if inverse == False: plt.imsave(name, depth, cmap='Spectral', vmin=min_depth, vmax=max_depth)

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

def compute_scale_and_shift_np(prediction, target, mask):
    """
    Compute optimal scale and shift for aligning predicted inverted depth to target.
    prediction, target, and mask are 2D NumPy arrays.
    """

    a00 = np.sum(mask * prediction * prediction)
    a01 = np.sum(mask * prediction)
    a11 = np.sum(mask)
    
    b0 = np.sum(mask * prediction * target)
    b1 = np.sum(mask * target)
    
    det = a00 * a11 - a01 * a01
    if det <= 0:
        scale = 1.0
        shift = 0.0
    else:
        scale = (a11 * b0 - a01 * b1) / det
        shift = (-a01 * b0 + a00 * b1) / det
    return scale, shift

def convert_inverted_to_depth_np(prediction, target, mask, depth_cap=10.0):
    """
    Converts predicted inverted relative depth (e.g. disparity) into absolute depth.
    prediction: 2D NumPy array (inverted relative depth).
    target: ground truth depth map (2D NumPy array).
    mask: binary mask (2D NumPy array) where valid pixels are 1.
    depth_cap: cap for the maximum depth (used to compute a minimum disparity).
    Returns:
      absolute_depth: 2D NumPy array of absolute depth.
    """
    # Create target disparity: valid pixels get 1.0 / target depth.
    target_disparity = np.zeros_like(target, dtype=np.float32)
    valid = (mask == 1)
    target_disparity[valid] = 1.0 / target[valid]
    
    # Compute optimal scale and shift from prediction to target disparity.
    scale, shift = compute_scale_and_shift_np(prediction, target_disparity, mask)
    
    # Align prediction.
    prediction_aligned = scale * prediction + shift
    
    # Cap the aligned disparity to avoid extremely small values.
    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap
    
    # Convert aligned disparity to absolute depth.
    absolute_depth = 1.0 / prediction_aligned
    return absolute_depth

def convert_relative_to_depth_np(prediction, target, mask, depth_cap=10.0):
    """
    Converts predicted relative depth into absolute depth.
    prediction: 2D NumPy array (relative depth).
    target: ground truth depth map (2D NumPy array).
    mask: binary mask (2D NumPy array) where valid pixels are 1.
    Returns:
      absolute_depth: 2D NumPy array of absolute depth.
    """
    # Compute optimal scale and shift from prediction to target disparity.
    scale, shift = compute_scale_and_shift_np(prediction, target, mask)
    
    # Align prediction.
    prediction_aligned = scale * prediction + shift
    
    return prediction_aligned

def evaluate_model_on_dataset(model, dataset, min_depth_eval=0, max_depth_eval=80.0, relative=True, inverse=True, max_images=None, save_output=False):
    images = dataset["images"]
    gt_depths = dataset["depths"]
    dataset_name = dataset["name"]
    preds = []
    errors_list = []

    # Slice dataset if max image count is set
    if max_images is not None:
        if max_images > len(images) or max_images <= 0:
            print(f"Max images was more than dataset count Max Images: {max_images} Image count: {len(images)}")
            return {}
        images = images[:max_images, :, :, :]
        gt_depths = gt_depths[:max_images, :, :]

  

    # Prediction for each image
    for image, gt in zip(images, gt_depths):
        pred_depth = model(image)
        preds.append(pred_depth)
        print(f"\rPredicted {len(preds)}/{len(images)} images", end="")
    
    pred_depths = np.stack(preds, axis=0)
    print(f"\nFinal shape of preds: Pred Depths: {pred_depths.shape}") 
    for image, gt_depth, pred_depth in zip(images, gt_depths, pred_depths):
        # Set infinite ground truth values to 0 (Will get cropped out by min_eval_depth)
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        # Mask for invalid values outside of min_depth_eval and max_depth_eval
        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        # if garg_crop:
        #     eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        # Removes borders
        if dataset_name == 'kitti':
            # Kitti eigen crop
            eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        elif dataset_name == "nyu":
            # Nyu specific kitti eigen crop
            eval_mask[45:471, 41:601] = 1
        else:
            eval_mask[:,:] = 1

        # Combine masks to remove cropped parts from image and depth
        valid_mask = np.logical_and(valid_mask, eval_mask)

        # Inverse Relative and Relative depth maps are converted to metric depth (mask is applied)
        if relative == True:
            if inverse == True:
                pred_depth_metric  = convert_inverted_to_depth_np(pred_depth, gt_depth, valid_mask, depth_cap=max_depth_eval)
            else:
                #pred_depth_metric = rel_depth_to_true_depth(pred_depth[valid_mask], gt_depth[valid_mask])
                pred_depth_metric = convert_relative_to_depth_np(pred_depth, gt_depth, valid_mask, depth_cap=max_depth_eval)
        else:
            pred_depth_metric = pred_depth[valid_mask]

        # Set invalid values to max and min
        pred_depth_metric[pred_depth_metric < min_depth_eval] = min_depth_eval
        pred_depth_metric[pred_depth_metric > max_depth_eval] = max_depth_eval
        pred_depth_metric[np.isinf(pred_depth_metric)] = max_depth_eval

        errors = compute_errors(gt_depth[valid_mask], pred_depth_metric[valid_mask])
        errors_list.append(errors)

        print("Min and max gt: ", gt_depth.min(), gt_depth.max())
        print("Min and max pred: ", pred_depth_metric.min(), pred_depth_metric.max())

        gt = gt_depth[valid_mask]

        if save_output and len(errors_list) <= save_output:
            save_depth(pred_depth, inverse=inverse, name=f"./output/{dataset_name}/output_depth{len(errors_list)}.png", max_depth=pred_depth.max(), min_depth=pred_depth.min())
            save_depth(np.where(valid_mask, pred_depth_metric, np.nan), name=f"./output/{dataset_name}/output_depth_masked{len(errors_list)}.png", max_depth=pred_depth_metric.max(), min_depth=pred_depth_metric.min())
            save_depth(np.where(valid_mask, gt_depth, np.nan), name=f"./output/{dataset_name}/output_depth_gt{len(errors_list)}.png", max_depth=gt.max(), min_depth=gt.min())
            save_image(image, name=f"./output/{dataset_name}/output_image{len(errors_list)}.png")

        print(f"\rProcessed {len(errors_list)}/{len(images)} images", end="")

    aggregated_errors = {k: np.mean([e[k] for e in errors_list]) for k in errors_list[0].keys()}
    return aggregated_errors

