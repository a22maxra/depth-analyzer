import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from image_helper import *
import scipy.io
matplotlib.use('TkAgg')

# # open output_image_mat.png
# img = plt.imread('output_image_mat.png')

# # shape of image
# print("Shape of image: ", img.shape)

with h5py.File('nyu_depth_v2_labeled.mat', 'r') as f:
    # Load datasets
    images = f['images']
    depths = f['depths']
    labels = f['labels']
    instances = f['instances']

    print(f"\n Keys in the file: {list(f.keys())} ")

    # Correct MATLAB's column-major order to (Height, Width, Channels, NumImages)
    images_arr = np.array(images)                       # Current shape: (1449, 3, 640, 480)
    images_arr = np.transpose(images_arr, (3, 2, 1, 0))
    print(f"\nRaw images shape: {images.shape}")        # (1449, 3, 640, 480)
    print("Corrected images shape:", images_arr.shape)  # Should be (480, 640, 3, 1449)

    depths_arr = np.array(depths)                       # Current shape: (1449, 640, 480)
    depths_arr = np.transpose(depths_arr, (2, 1, 0))
    print(f"\nRaw depths shape: {depths.shape}")        # (1449, 640, 480)
    print("Corrected depths shape:", depths_arr.shape)  # Should be (480, 640, 1449)

    crop = 16
    images_cropped = images_arr[crop:-crop, crop:-crop, :, :]
    depths_cropped = depths_arr[crop:-crop, crop:-crop, :]

    scipy.io.savemat('nyu_depth_v2_cropped.mat', {
    'images': images_cropped,
    'depths': depths_cropped,
    # include other variables as needed (e.g., 'labels', 'instances')
    })