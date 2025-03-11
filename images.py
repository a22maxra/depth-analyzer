import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from image_helper import *
from scipy.io import loadmat
from model_wrapper import *
from load_kitti_eigen_test import load_kitti
from load_diode_val import load_diode
from load_scannet_val import load_scannet
from load_space_val import load_in_space_type
from load_eth3d_test import load_eth3d


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
    elif args.model == "marigold":
        inverse = False
        relative = True
    elif args.model == "depthanythingv2" or args.model == "midas":
        inverse = True
        relative = True

    if args.dataset == "kitti":
        dataset = load_kitti(args.max)
        dataset["name"] = "kitti"
        max_depth = 80.0
        min_depth = 0.001
    elif args.dataset == "nyu":
        dataset = load_mat_dataset("datasets/nyu/nyu_depth_v2_processed")
        dataset["name"] = "nyu"
        dataset["images"] = dataset["images"].transpose(3, 0, 1, 2) # Should be (N, H, W, C)
        dataset["depths"] = dataset["depths"].transpose(2, 0, 1)
        max_depth = 10.0
        min_depth = 0.001
    elif args.dataset == "diodeout":
        dataset = load_diode(args.max, scene_type="outdoor")
        dataset["name"] = "diode"
        max_depth = 80.0
        min_depth = 0.001
    elif args.dataset == "diodein":
        dataset = load_diode(args.max,  scene_type="indoors")
        dataset["name"] = "diode"
        max_depth = 20.0
        min_depth = 0.001
    elif args.dataset == "diode":
        dataset = load_diode(args.max)
        dataset["name"] = "diode"
        max_depth = 350.0
        min_depth = 0.001
    elif args.dataset == "scannet":
        dataset = load_scannet(args.max)
        dataset["name"] = "scannet"
        max_depth = 10.0
        min_depth = 0.001
    elif args.dataset == "space":
        dataset = load_in_space_type(args.max)
        dataset["name"] = "space"
        max_depth = 20.0
        min_depth = 0.001
    elif args.dataset == "eth3d":
        dataset = load_eth3d(args.max)
        dataset["name"] = "eth3d"
        max_depth = 75.0
        min_depth = 0.001

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
    metrics = evaluate_model_on_dataset(model_fn, dataset, min_depth_eval=min_depth, max_depth_eval=max_depth, relative=relative, inverse=inverse, max_images=args.max, save_output=args.save)
    
    print("\nAggregated error metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()