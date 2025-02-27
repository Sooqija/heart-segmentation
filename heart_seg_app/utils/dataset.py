import os
import numpy as np

import torch
import torch.utils.data

from monai.transforms import MapTransform

def collect_data_paths(image_dir, label_dir, prefix="", postfix="") -> list:
    """
    Collects image and label file paths from the specified directories.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): Directory containing label files.
        prefix (str, optional): Prefix to filter files. Defaults to "".
        postfix (str, optional): Postfix to filter files. Defaults to "".

    Returns:
        list: List of dictionaries with "image" and "label" keys.
    """
    data = []
    image_paths = sorted(
        [os.path.relpath(os.path.join(image_dir, path), start=os.getcwd()) for path in os.listdir(image_dir) 
         if path.startswith(prefix) and path.endswith(f"_image.nii{postfix}")]
    )
    label_paths = sorted(
        [os.path.relpath(os.path.join(label_dir, path), start=os.getcwd()) for path in os.listdir(label_dir) 
         if path.startswith(prefix) and path.endswith(f"_label.nii{postfix}")]
    )
    
    for image, label in zip(image_paths, label_paths):
        data.append({"image": image, "label": label})
    
    return data

def split_dataset(data, split_ratios=(0.75, 0.2, 0.05), seed=0) -> dict:
    """
    Splits dataset into training, validation, and test sets based on given ratios.
    
    Args:
        data (list): List of dataset entries.
        split_ratios (tuple, optional): Tuple containing split ratios for train, val, and test. Defaults to (0.75, 0.2, 0.05).
        seed: Random seed for reproducibility

    Returns:
        dict: Dictionary with "train", "val", and "test" keys mapping to dataset subsets.
    """
    train_size = int(split_ratios[0] * len(data))
    val_size = int(split_ratios[1] * len(data))
    test_size = len(data) - train_size - val_size
    
    if seed is not None:
        torch.manual_seed(seed)
    train_data, val_data, test_data = torch.utils.data.random_split(data, [train_size, val_size, test_size])
    
    return {
        "train": [data[i] for i in train_data.indices],
        "val": [data[i] for i in val_data.indices],
        "test": [data[i] for i in test_data.indices]
    }

class ToOneHotd(MapTransform):
    """
    MONAI-compatible transformation for label preprocessing.
    Converts label values into one-hot representation.
    """

    def __init__(self, keys, label_values):
        """
        Args:
            keys (list): List of keys to apply the transformation to (e.g., ["label"]).
            label_values (list): List of label values for one-hot encoding.
        """
        super().__init__(keys)
        self.label_values = label_values

    def __call__(self, data):
        """
        Applies the transformation.

        Args:
            data (dict): A dictionary containing `keys` (usually "label") with the label image.

        Returns:
            dict: An updated dictionary with the transformed labels.
        """
        d = dict(data)
        for key in self.keys:
            label = d[key]
            values = self.label_values
            processed_label = torch.zeros((len(values), *label.shape), dtype=torch.float32)
            
            for i, value in enumerate(values):
                processed_label[i] = torch.where(label == value, 1.0, 0.0)

            d[key] = processed_label
        
        return d

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None, prefix="", postfix=""):

        self.inputs_paths = []
        self.targets_paths = []

        for path in sorted(os.listdir(image_dir)):
            if path.startswith(prefix) and path.endswith(f"_image.nii{postfix}"):
                self.inputs_paths.append(os.path.join(image_dir, path))
                
        for path in sorted(os.listdir(label_dir)):
            if path.startswith(prefix) and path.endswith(f"_label.nii{postfix}"):
                self.targets_paths.append(os.path.join(label_dir, path))

        if len(self.inputs_paths) != len(self.targets_paths):
            raise ValueError(f"Inputs and targets have different lengths: {len(self.inputs_paths)} vs {len(self.targets_paths)}")
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs_paths)
    
    def __getitem__(self, idx):
        image_path = self.inputs_paths[idx]
        label_path = self.targets_paths[idx]

        image : np.ndarray = np.load(image_path)
        label : np.ndarray = np.load(label_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    

class ImagePreprocessing(object):
    def __init__(self, mode="norm", **kwargs):
        """
        mode ["norm", "to mean-std"]
        """
        self.mode = mode
        # self.src_mean, self.src_std, self.dst_mean, self.dst_std = kwargs
    
    def __call__(self, image : np.ndarray):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}.")
        
        if self.mode == "norm":
            image = (image - image.mean()) / (image.std() + 1e-8)
        if self.mode == "to mean-std":
            image = (image - self.src_mean) / (self.src_std + 1e-8) * self.dst_mean + self.dst_std
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        # image = torch.from_numpy(image).type(torch.FloatTensor)
        return image

class LabelPreprocessing(object):
    def __init__(self, label_values):
        self.label_values = label_values

    def __call__(self, label):
        values = self.label_values
        processed_label = np.zeros(shape=(len(values), *label.shape), dtype=np.float32)
        for i, value in enumerate(values):
            processed_label[i] = np.where(label == value, 1.0, 0.0)
        # processed_label = np.expand_dims(processed_label, axis=0)
        processed_label = torch.from_numpy(processed_label).type(torch.FloatTensor)
        return processed_label
    

def label_postprocessing(label: torch.Tensor):
    label = torch.softmax(label, dim=1).argmax(dim=1).squeeze(0)
    label_values = list(range(8))
    processed_label = torch.zeros(size=(len(label_values), *label.shape), dtype=torch.int32, device="cuda")
    for i, value in enumerate(label_values):
        processed_label[i] = torch.where(label == value, 1, 0)

    return processed_label