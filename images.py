import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from image_helper import *
from scipy.io import loadmat
from model_wrapper import *
import scipy.io

# Use an interactive backend
matplotlib.use('TkAgg')


# img = extract_image(images_arr, index = 1)
# save_image(img)

# depth_abs = extract_depth(depths_arr, index = 1)
# save_depth(depth_abs)

# # Load depth map from MDE model (saved as single depth map in .mat file) 
# data = loadmat('MDE_depth_output.mat')
# depth_rel = data['depth'] 
# print(f"\nShape of depth_rel map array: {depth_rel.shape}, Min value: {depth_rel.min()}, Max value: {depth_rel.max()}")

# # Translate inverse relative depth_abs to absolute depth_abs
# depth_rel_to_abs = inverse_rel_depth_to_true_depth(depth_rel, depth_abs)

# # Compute Mean Absolute Error
# mae = compute_absolute_error(depth_rel_to_abs, depth_abs)

# print(f"\n\n\nMean Absolute Error: {mae}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MDE Model on .mat Dataset")
    parser.add_argument("--mat-path", type=str, required=True,
                        help="Path to the .mat file containing images and depths")
    parser.add_argument("--input-size", type=int, default=518, help="Input size for the model")
    parser.add_argument("--model", type=str, default="depth_anything_v2", help="Model name")
    parser.add_argument("--encoder", type=str, default="vits", help="Encoder type (vits, vitb, vitl, vitg)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device, encoder_choice=args.encoder)

    # Combine them into a list of (image, depth) pairs
    dataset = load_mat_dataset(args.mat_path)
    
    # Create a callable for the model.
    model_fn = lambda img: model_callable(img, args.input_size, model)
    
    # Evaluate the model on the dataset.
    metrics = evaluate_model_on_dataset(model_fn, dataset, do_convert=True)
    
    print("Aggregated error metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()