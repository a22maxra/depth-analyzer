import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from image_helper import *
from scipy.io import loadmat

# Use an interactive backend
matplotlib.use('TkAgg')

# Load datasets
images_arr = load_mat_file_images(file_path = 'nyu_depth_v2_cropped.mat')
depths_arr = load_mat_file_depths(file_path = 'nyu_depth_v2_cropped.mat')   

img = extract_image(images_arr, index = 1)
save_image(img)

depth_abs = extract_depth(depths_arr, index = 1)
save_depth(depth_abs)

# Load depth map from MDE model (saved as single depth map in .mat file) 
data = loadmat('MDE_depth_output.mat')
depth_rel = data['depth'] 
print(f"\nShape of depth_rel map array: {depth_rel.shape}, Min value: {depth_rel.min()}, Max value: {depth_rel.max()}")

# Translate inverse relative depth_abs to absolute depth_abs
depth_rel_to_abs = inverse_rel_depth_to_true_depth(depth_rel, depth_abs)

# Compute Mean Absolute Error
mae = compute_absolute_error(depth_rel_to_abs, depth_abs)

print(f"\n\n\nMean Absolute Error: {mae}")