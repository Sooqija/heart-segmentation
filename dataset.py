import os
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None, prefix="", postfix=""):

        self.inputs_paths = []
        self.targets_paths = []

        for path in os.listdir(image_dir):
            if path.startswith(prefix) and path.endswith(f"_image.nii.npy{postfix}"):
                self.inputs_paths.append(os.path.join(image_dir, path))
                
        for path in os.listdir(label_dir):
            if path.startswith(prefix) and path.endswith(f"_label.nii.npy{postfix}"):
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
    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).type(torch.FloatTensor)
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
    label = torch.softmax(label, dim=1).argmax(dim=1).to("cpu").squeeze(0)
    label_values = list(range(8))
    processed_label = torch.zeros(size=(len(label_values), *label.shape), dtype=torch.int32)
    for i, value in enumerate(label_values):
        processed_label[i] = torch.where(label == value, 1, 0)

    return processed_label