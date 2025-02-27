import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from image_helper import *
from scipy.io import loadmat
from model_wrapper import *
from load_kitti_eigen_test import *


def main():
    parser = argparse.ArgumentParser(description="Evaluate MDE Model on .mat Dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the .mat file containing images and depths")
    parser.add_argument("--model", type=str, default="Default", help="Model name")
    parser.add_argument("--encoder", type=str, default="Default", help="Encoder type (vits, vitb, vitl, vitg)")
    parser.add_argument("--max", type=int, default=None, help="max image count to evaluate")
    parser.add_argument("--save", type=int, default=False, help="Save x amount of images and depths")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device, encoder_choice=args.encoder)

    if args.model == "depthpro" or args.model == "zoedepth":
        inverse = False
        relative = False
    elif args.model == "midas":
        inverse = False
        relative = True
    elif args.model == "depthanythingv2" or args.model == "midas":
        inverse = True
        relative = True

    if args.dataset == "kitti":
        dataset = load_kitti()
    elif args.dataset == "nyu":
        dataset = load_mat_dataset("nyu_depth_v2_cropped")
        dataset["images"] = dataset["images"].transpose(3, 0, 1, 2) # Should be (N, H, W, C)
        dataset["depths"] = dataset["depths"].transpose(2, 0, 1)

    # Print run settings
    print(f"\nModel: {args.model}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {device}")
    print(f"Max images: {args.max}")
    print(f"Save output: {args.save}")
    print(f"Inverse depth: {inverse}")
    print(f"Relative depth: {relative}")
    
    # Create a callable for the model.
    model_fn = lambda img: model_callable(img, model)
    
    # Evaluate the model on the dataset.
    metrics = evaluate_model_on_dataset(model_fn, dataset, max_images=args.max, save_output=args.save, inverse=inverse, relative=relative)
    
    print("Aggregated error metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()