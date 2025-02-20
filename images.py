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


def main():
    parser = argparse.ArgumentParser(description="Evaluate MDE Model on .mat Dataset")
    parser.add_argument("--mat-path", type=str, required=True,
                        help="Path to the .mat file containing images and depths")
    parser.add_argument("--model", type=str, default="depth_anything_v2", help="Model name")
    parser.add_argument("--encoder", type=str, default="vits", help="Encoder type (vits, vitb, vitl, vitg)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device, encoder_choice=args.encoder)

    dataset = load_mat_dataset(args.mat_path)
    
    # Create a callable for the model.
    model_fn = lambda img: model_callable(img, model)
    
    # Evaluate the model on the dataset.
    metrics = evaluate_model_on_dataset(model_fn, dataset, do_convert=True)
    
    print("Aggregated error metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()