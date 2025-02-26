import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from image_helper import *
from scipy.io import loadmat
from model_wrapper import *


def main():
    parser = argparse.ArgumentParser(description="Evaluate MDE Model on .mat Dataset")
    parser.add_argument("--mat-path", type=str, required=True,
                        help="Path to the .mat file containing images and depths")
    parser.add_argument("--model", type=str, default="Default", help="Model name")
    parser.add_argument("--encoder", type=str, default="Default", help="Encoder type (vits, vitb, vitl, vitg)")
    parser.add_argument("--max", type=int, default=None, help="max image count to evaluate")
    parser.add_argument("--save", type=int, default=False, help="Save x amount of images and depths")
    parser.add_argument("--inverse", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set if model produces inverse depth")
    parser.add_argument("--relative", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set if model produces relative depth")
    args = parser.parse_args()
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device, encoder_choice=args.encoder)

    dataset = load_mat_dataset(args.mat_path)

    # Print run settings
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {device}")
    print(f"Max images: {args.max}")
    print(f"Save output: {args.save}")
    print(f"Inverse depth: {args.inverse}")
    print(f"Relative depth: {args.relative}")
    
    # Create a callable for the model.
    model_fn = lambda img: model_callable(img, model)
    
    # Evaluate the model on the dataset.
    metrics = evaluate_model_on_dataset(model_fn, dataset, max_images=args.max, save_output=args.save, inverse=args.inverse, relative=args.relative)
    
    print("Aggregated error metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()