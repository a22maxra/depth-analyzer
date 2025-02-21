from image_helper import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
import cv2
from image_helper import *

# Define constants
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0
EPSILON = 1e-6  # small constant to avoid division by zero

def calculate_abs_rel(gt_depth, pred_depth):
    """
    Computes the Absolute Relative Error (AbsRel) between ground truth and predicted depth.
    
    Args:
        gt_depth (numpy array): Ground truth depth map.
        pred_depth (numpy array): Predicted depth map.
        
    Returns:
        float: Absolute relative error.
    """
    abs_rel_error = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
    return abs_rel_error

def find_scale_shift(gt, pred):
    """Compute optimal scale (s) and shift (b) using least squares."""
    A = np.vstack([pred.flatten(), np.ones_like(pred.flatten())]).T
    s, b = np.linalg.lstsq(A, gt.flatten(), rcond=None)[0]
    return s, b

def apply_scale_shift(pred, s, b):
    """Apply scale and shift to predicted depth map."""
    # s (scale) and b (shift)
    return s * pred + b


# Load ground truth data from the NYU Depth V2 cropped .mat file.
images_arr = load_mat_file_images("nyu_depth_v2_cropped_10.mat")
depths_arr = load_mat_file_depths("nyu_depth_v2_cropped_10.mat")

# Extract one image and its corresponding ground truth depth.
img = extract_image(images_arr, index=1)
depth_abs = extract_depth(depths_arr, index=1)  # ground truth absolute depth

# Load the predicted inverse depth from MDE_depth_output.mat.
mat_data = loadmat('MDE_depth_output.mat')
# Here we assume 'depth' holds the predicted inverse depth (after using a ReLU-like output).
inv_rel_depth = mat_data['depth']

# Create a mask to consider only valid ground truth depth values.
mask = np.logical_and(depth_abs >= MIN_DEPTH, depth_abs <= MAX_DEPTH)
gt_valid = depth_abs[mask]
pred_valid = inv_rel_depth[mask]

# Create a copy of the valid ground truth and predicted depth values.
pred_valid_copy = pred_valid.copy()
gt_valid_copy = gt_valid.copy()

pred_valid_copy_2 = pred_valid.copy()
gt_valid_copy_2 = gt_valid.copy()

# Apply median scaling: adjust predicted depth so that its median matches the ground truth.
scaling_ratio = np.median(gt_valid) / np.median(pred_valid)
pred_valid_scaled = pred_valid * scaling_ratio

# Compute and print the Absolute Relative Error.
abs_rel_error = np.mean(np.abs(gt_valid - pred_valid_scaled) / gt_valid)
print("Absolute Relative Error (abs_rel):", abs_rel_error, " (probably way off)")

# Alternative method to compute the Absolute Relative Error.
pred_valid_copy = inverse_rel_depth_to_true_depth(pred_valid_copy, gt_valid_copy)
abs_rel_error = np.mean(np.abs(gt_valid_copy - pred_valid_copy) / gt_valid_copy)
print("Absolute Relative Error (abs_rel):", abs_rel_error, " (probably more correct)")

abs_rel_error = calculate_abs_rel(gt_valid_copy_2, pred_valid_copy_2)
print("Absolute Relative Error (abs_rel):", abs_rel_error, " (trying other way)")