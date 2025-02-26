import argparse
import glob
import os
import torch
import numpy as np
from image_helper import *
import sys
import tempfile
from PIL import Image



# Determine the project root (assuming helper/ is directly under /home/max/code)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Now add the folder that contains depth_anything_v2. In your case, it's in Depth-Anything-V2.
depthanything_dir = os.path.join(project_root, 'Depth-Anything-V2')
if depthanything_dir not in sys.path:
    sys.path.insert(0, depthanything_dir)
midas_dir = os.path.join(project_root, 'MiDaS')
if midas_dir not in sys.path:
    sys.path.insert(0, midas_dir)

# Factory function to load a model based on the given name.
def load_model(model_name, device, encoder_choice='vitl'):
    """
    Dynamically load a model based on the given name. Currently, only 'depth_anything_v2'
    is implemented.
    """
    if model_name == "depth_anything_v2":
        print("Loading Depth-Anything-V2 model with encoder:", encoder_choice)
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

    if model_name == "midas":
        print("Loading MiDaS model with encoder:", encoder_choice)
        from midas.model_loader import load_model as midas_load_model
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(project_root, 'MiDaS', 'weights', f'{encoder_choice}.pt')
        if model_path is None:
            raise ValueError(f"MiDaS model {encoder_choice} not recognized.")
        # Load MiDaS model and its associated transform and size parameters.
        model, transform, net_w, net_h = midas_load_model(device, model_path, encoder_choice)
        # Return as a dictionary so we can later branch in inference.
        return {"model": model, "transform": transform, "net_w": net_w, "net_h": net_h, "type": "midas"}

    if model_name == "depthpro":
        print("Loading Depth-PRO model")
        import depth_pro
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        depthpro_dir = os.path.join(project_root, 'ml-depth-pro')
        if depthpro_dir not in sys.path:
            sys.path.insert(0, depthpro_dir)

        os.chdir(depthpro_dir)

        model, transform = depth_pro.create_model_and_transforms()
        model.eval()

        return {"model": model, "transform": transform, "type": "depthpro"}
    
    if model_name == "marigold":
        print("Loading Marigold model")
        import diffusers
        import torch
        pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")
        return {"model": pipe, "type": "marigold"}

    else:
        raise ValueError(f"Model {model_name} not implemented.")

def get_relative_depth(image, model):
    """
    Preprocess the image and call the model's inference method.
    Returns the relative depth map as a 2D numpy array.
    For depth_anything_v2, we assume the model implements infer_image().
    For MiDaS, we use the provided transform and forward pass.
    """
    # MiDaS branch: model is a dict with key "type" == "midas"
    if isinstance(model, dict) and model.get("type") == "midas":
        loaded = model
        # MiDaS expects a float image in [0,1]
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        # Apply MiDaS transform
        transformed = loaded["transform"]({"image": image})
        input_image = transformed["image"]
        # Convert to torch tensor and add batch dimension.
        device = next(loaded["model"].parameters()).device
        sample = torch.from_numpy(input_image).to(device).unsqueeze(0)
        with torch.no_grad():
            prediction = loaded["model"].forward(sample)
            # Interpolate prediction to original image size.
            target_size = (image.shape[1], image.shape[0])  # (width, height)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        return prediction    
    
    elif isinstance(model, dict) and model.get("type") == "depthpro":
        import depth_pro

        pil_image = Image.fromarray(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            pil_image.save(temp_path)

        loaded = model
        tranform = loaded["transform"]
        model = loaded["model"]

        processed_image, _, f_px = depth_pro.load_rgb(temp_path)
        processed_image = tranform(processed_image)

        prediction = model.infer(processed_image, f_px=f_px)
        depth = prediction["depth"]
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()

        os.remove(temp_path)

        return depth
    
    elif isinstance(model, dict) and model.get("type") == "marigold":
        model = model["model"]
        pil_image = Image.fromarray(image)

        depth = model(pil_image)
        depth = depth.prediction
        depth = depth.transpose(1, 2, 0, 3)
        depth = np.squeeze(depth)

        return depth
    
    else:
        # depth_anything_v2 branch
        return model.infer_image(image)

def model_callable(raw_image, model):
    """
    This function wraps get_relative_depth so it can be passed to the evaluation function.
    It takes an image and returns the relative depth map.
    """
    return get_relative_depth(raw_image, model)
