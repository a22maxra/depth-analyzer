import os
import tarfile
import numpy as np
from PIL import Image

# Assume this enum or constant is defined as needed.
class DepthFileNameMode:
    id = 0

# Base class (simplified, assuming you have something like this)
class BaseDepthDataset:
    def __init__(self, min_depth, max_depth, has_filled_depth, name_mode, **kwargs):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.has_filled_depth = has_filled_depth
        self.name_mode = name_mode
        self.dataset_dir = kwargs.get("dataset_dir", "")
        self.file_list = kwargs.get("file_list", [])
        self.is_tar = kwargs.get("is_tar", False)
        self.tar_obj = None

# ETH3D dataset loader using tarfile (adapted from the provided code)
class ETH3DDataset(BaseDepthDataset):
    # Expected resolution (adjust if necessary)
    HEIGHT, WIDTH = 4032, 6048

    def __init__(self, **kwargs) -> None:
        super().__init__(
            min_depth=1e-5,
            max_depth=np.inf,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        # Read special binary data from tar archive
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            # Prepend "./" or an appropriate prefix if your tar archive contains a top-level folder.
            full_path = "./" + rel_path
            try:
                member = self.tar_obj.getmember(full_path)
            except KeyError:
                raise FileNotFoundError(f"Depth file {full_path} not found in tar archive.")
            binary_data = self.tar_obj.extractfile(member).read()
        else:
            depth_path = os.path.join(self.dataset_dir, rel_path)
            with open(depth_path, "rb") as file:
                binary_data = file.read()
        # Convert binary data to a NumPy array of float32 and reshape
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()
        depth_decoded[depth_decoded == np.inf] = 0.0
        depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH))
        return depth_decoded

    def _read_image_file(self, rel_path):
        # Read RGB image from tar archive
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            full_path = "./" + rel_path  # adjust prefix as needed
            try:
                member = self.tar_obj.getmember(full_path)
            except KeyError:
                raise FileNotFoundError(f"Image file {full_path} not found in tar archive.")
            with self.tar_obj.extractfile(member) as file_obj:
                image = Image.open(file_obj).convert("RGB")
        else:
            image_path = os.path.join(self.dataset_dir, rel_path)
            image = Image.open(image_path).convert("RGB")
        return np.array(image)

    def __getitem__(self, index):
        # Get relative paths from the file_list
        img_rel_path, depth_rel_path = self.file_list[index]
        image = self._read_image_file(img_rel_path)
        depth = self._read_depth_file(depth_rel_path)
        return {"image": image, "depth": depth}

    def __len__(self):
        return len(self.file_list)

# Function to create file_list from a text file
def create_eth3d_file_list(txt_file):
    file_list = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_path, depth_path = parts[0], parts[1]
        file_list.append((img_path, depth_path))
    return file_list

# The load_eth3d function that returns the dataset dictionary
def load_eth3d(max_images=None):
    txt_file = './datasets/eth3d/eth3d_filename_list.txt'
    base_dir = './datasets/eth3d/'
    tar_file = os.path.join(base_dir, "eth3d.tar")
    
    file_list = create_eth3d_file_list(txt_file)
    
    # Initialize the dataset to use the tar archive.
    dataset = ETH3DDataset(
        dataset_dir=tar_file,
        file_list=file_list,
        is_tar=True
    )
    
    images_list = []
    depths_list = []
    for idx in range(len(dataset)):
        if max_images is not None and idx >= max_images:
            break
        sample = dataset[idx]
        images_list.append(sample["image"])
        depths_list.append(sample["depth"])
        print(f"\rLoaded {idx+1}/{len(dataset)} samples", end="")
    
    images = np.stack(images_list, axis=0)   # shape: (N, H, W, C)
    depths = np.stack(depths_list, axis=0)     # shape: (N, H, W)
    
    print("\nShape of depths array:", depths.shape)
    print("Shape of images array:", images.shape)
    
    zipped_dataset = {'images': images, 'depths': depths, 'name': 'eth3d'}
    return zipped_dataset
