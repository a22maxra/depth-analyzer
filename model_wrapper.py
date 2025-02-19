import argparse
import cv2
import glob
import os
import torch
import numpy as np
from image_helper import *
import sys

# Determine the project root (assuming helper/ is directly under /home/max/code)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Now add the folder that contains depth_anything_v2. In your case, it's in Depth-Anything-V2.
depthanything_dir = os.path.join(project_root, 'Depth-Anything-V2')
if depthanything_dir not in sys.path:
    sys.path.insert(0, depthanything_dir)

# Factory function to load a model based on the given name.
def load_model(model_name, device, encoder_choice='vitl'):
    """
    Dynamically load a model based on the given name. Currently, only 'depth_anything_v2'
    is implemented.
    """
    if model_name == "depth_anything_v2":
        from depth_anything_v2.dpt import DepthAnythingV2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        model = DepthAnythingV2(**model_configs[encoder_choice])
        # Determine the project root; assuming model_wrapper.py is in /home/max/code/helper/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Build the absolute path to the checkpoint inside the Depth-Anything-V2 folder.
        checkpoint_path = os.path.join(project_root, 'Depth-Anything-V2', 'checkpoints', f'depth_anything_v2_{encoder_choice}.pth')
        print("Loading checkpoint from:", checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(device).eval()
        return model
    else:
        raise ValueError(f"Model {model_name} not implemented.")

def get_relative_depth(image, model):
    """
    Preprocess the image and call the model's inference method.
    Returns the relative depth map as a 2D numpy array.
    """
    # Call the model's inference function (assumed to be infer_image)
    depth = model.infer_image(image)
    return depth

def model_callable(raw_image, model):
    """
    This function wraps get_relative_depth so it can be passed to the evaluation function.
    It takes an image and returns the relative depth map.
    """
    return get_relative_depth(raw_image, model)
