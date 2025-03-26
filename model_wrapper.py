import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from image_helper import *
import sys
import tempfile
from PIL import Image

# Factory function to load a model based on the given name.
def load_model(model_name, device, encoder_choice='vitl', epoch=5):
    """
    Dynamically load a model based on the given name. Currently, only 'depth_anything_v2'
    is implemented.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if model_name == "depthanythingv2":
        print("Loading Depth-Anything-V2 model with encoder:", encoder_choice)
        depthanything_dir = os.path.join(project_root, 'Depth-Anything-V2')
        if depthanything_dir not in sys.path:
            sys.path.insert(0, depthanything_dir)
        from depth_anything_v2.dpt import DepthAnythingV2
        import torch
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        model = DepthAnythingV2(**model_configs[encoder_choice])
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Build the absolute path to the checkpoint inside the Depth-Anything-V2 folder.
        checkpoint_path = os.path.join(project_root, 'Depth-Anything-V2', 'checkpoints', f'depth_anything_v2_{encoder_choice}.pth')
        print("Loading checkpoint from:", checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(device).eval()
        return {"model": model, "type": "depthanythingv2"}

    if model_name == "dametric":
        print("Loading metric (finetuned on KITTI) Depth-Anything-V2 model with encoder:", encoder_choice)
        dametric_dir = os.path.join(project_root, 'Depth-Anything-V2', 'metric_depth')
        if dametric_dir not in sys.path:
            sys.path.insert(0, dametric_dir)
        from depth_anything_v2.dpt import DepthAnythingV2
        import torch
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        model = DepthAnythingV2(**{**model_configs[encoder_choice], 'max_depth': 80.0})
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Build the absolute path to the checkpoint inside the Depth-Anything-V2 folder.
        checkpoint_path = os.path.join(project_root, 'Depth-Anything-V2', 'metric_depth', 'exp', 'kitti', f'checkpoint_epoch_{epoch}.pth')
        print("Loading checkpoint from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        my_state_dict = {}
        for key in checkpoint['model'].keys():
            my_state_dict[key.replace('module.', '')] = checkpoint['model'][key]
        model.load_state_dict(my_state_dict)
        model = model.to(device).eval()
        return {"model": model, "type": "dametric"}

    if model_name == "midas":
        print("Loading MiDaS model with encoder:", encoder_choice)
        midas_dir = os.path.join(project_root, 'MiDaS')
        if midas_dir not in sys.path:
            sys.path.insert(0, midas_dir)
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
        import torch
        from depth_pro.depth_pro import DepthProConfig
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        depthpro_dir = os.path.join(project_root, 'ml-depth-pro', 'src')
        if depthpro_dir not in sys.path:
            sys.path.insert(0, depthpro_dir)
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri="./ml-depth-pro/checkpoints/depth_pro.pt",
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
        model.eval()

        return {"model": model, "transform": transform, "type": "depthpro"}

    if model_name == "marigold":
        print("Loading Marigold model")
        import diffusers
        import torch
        pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
        ).to(device)
        return {"model": pipe, "type": "marigold"}

    if model_name == "zoedepth":
        from transformers import pipeline

        pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")
        return {"model": pipe, "type": "zoedepth"}

    if model_name == "zoeft":
        zoe_dir = os.path.join(project_root, 'ZoeDepth')
        if zoe_dir not in sys.path:
            sys.path.insert(0, zoe_dir)
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config

        dataset = "kitti"
        pretrained_resource=f"local::/mnt/mh_grupp/ZoeDepth/trained_checkpoints/ZoeDepthv1_25-Mar_15-28-d5a0bf3b51e0_epoch_{epoch}.pt"
        overwrite = {"pretrained_resource": pretrained_resource}

        config = get_config("zoedepth", "eval", dataset, **overwrite)
        model_zoe_ft = build_model(config)
        model_zoe_ft.to(device).eval()
        return {"model": model_zoe_ft, "type": "zoeft"}

    if model_name == "unidepthv1":
        unidepth_dir = os.path.join(project_root, 'UniDepth')
        if unidepth_dir not in sys.path:
            sys.path.insert(0, unidepth_dir)
        import torch
        from unidepth.models import UniDepthV1
        model_id = f"unidepth-v1-vit{encoder_choice}14"
        print(f"Loading UniDepthV1 model from: lpiccinelli/{model_id}")
        model = UniDepthV1.from_pretrained(f"lpiccinelli/{model_id}")
        model = model.to(device).eval()
        return {"model": model, "type": "unidepthv1"}

    if model_name == "unidepthv2":
        unidepth_dir = os.path.join(project_root, 'UniDepth')
        if unidepth_dir not in sys.path:
            sys.path.insert(0, unidepth_dir)
        import torch
        from unidepth.models import UniDepthV2
        model_id = f"unidepth-v2-vit{encoder_choice}14"
        print(f"Loading UniDepthV2 model from: lpiccinelli/{model_id}")
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_id}")
        model.interpolation_mode = "bilinear"
        model = model.to(device).eval()
        return {"model": model, "type": "unidepthv2"}

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

    elif isinstance(model, dict) and model.get("type") == "depthanythingv2" or model.get("type") == "dametric":
        model = model["model"]
        return model.infer_image(image)

    elif isinstance(model, dict) and model.get("type") == "zoedepth":
        model = model["model"]
        pil_image = Image.fromarray(image)
        depth = model(pil_image)
        depth = depth["predicted_depth"]
        depth = np.array(depth)
        return depth

    elif isinstance(model, dict) and model.get("type") == "zoeft":
        model = model["model"]
        device = next(model.parameters()).device

        # Store original size for later interpolation
        original_size = (image.shape[0], image.shape[1])  # (H, W)
        
        # Preprocess image
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.transpose(image, (0, 3, 1, 2))  # NHWC -> NCHW
        image = torch.from_numpy(image).float().to(device)
        print("Image shape:", image.shape)

        with torch.no_grad():
            depth = model(image)
            depth = depth["metric_depth"]  # Shape: (B, H, W)

            if depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            elif depth.ndim == 3:
                depth = depth.unsqueeze(1)  # Shape: (B, 1, H, W)
            depth = nn.functional.interpolate(
                depth, (original_size[0], original_size[1]), mode='bilinear', align_corners=True)

            # Move to CPU and convert to numpy
            depth = depth.cpu().detach().numpy().squeeze()

        return depth

    elif isinstance(model, dict) and model.get("type") == "unidepthv1":
        model = model["model"]
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        predictions = model.infer(rgb)
        depth = predictions["depth"].squeeze().cpu().numpy()
        return depth

    elif isinstance(model, dict) and model.get("type") == "unidepthv2":
        model = model["model"]
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        predictions = model.infer(rgb)
        depth = predictions["depth"].squeeze().cpu().numpy()
        return depth

def model_callable(raw_image, model):
    """
    This function wraps get_relative_depth so it can be passed to the evaluation function.
    It takes an image and returns the relative depth map.
    """
    return get_relative_depth(raw_image, model)
