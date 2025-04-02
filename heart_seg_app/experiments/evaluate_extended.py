import torch

from monai.data.dataset import Dataset
from monai.data import DataLoader, NumpyReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, NormalizeIntensityd,
    MapTransform,
)
from monai.networks.nets import UNETR

from heart_seg_app.utils.config import load_config
from heart_seg_app.utils.dataset import (
    ToOneHotd,
    collect_data_paths, label_postprocessing)
from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.utils.visualization import make_grid_image
from heart_seg_app.models.unetr import unetr

import torchvision.utils

import os
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

class ToOneHotExtendedd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            processed_label = torch.zeros((len(d["label_map"].keys()), *label.shape), dtype=torch.float32)
            for i, meta_data in enumerate(d["label_map"].values()):
                value = meta_data["value"]
                for v in value:
                    processed_label[i] = torch.logical_or(processed_label[i], torch.where(label == v, 1.0, 0.0))
            d[key] = processed_label
        
        return d

def create_active_classes_mask(label_map : dict):
    active_classes_mask = torch.zeros(len(label_map.keys()))

    for i, class_meta in enumerate(label_map.values()):
        if len(class_meta["value"]):
            active_classes_mask[i] = 1
            
    return active_classes_mask

def create_custom_cmap_extended(label_map : dict):
    pixel_values = [np.array(value["value"][0])[0] for value in label_map.values() if len(value["value"])]
    colors = [value["color"][0] for value in label_map.values()]
    
    if len(pixel_values) < 3:
        return matplotlib.colors.ListedColormap(list(colors), N=len(colors)).with_extremes(under='black', over='white'), None
    
    mid_bounds = [(pixel_values[i] + pixel_values[i+1]) / 2 for i in range(len(pixel_values)-1)]
    custom_cmap = matplotlib.colors.ListedColormap(list(colors)[1:], N=len(colors)).with_extremes(under='black', over='green')
    norm = matplotlib.colors.BoundaryNorm(mid_bounds, custom_cmap.N - 2)
    return custom_cmap, norm

def apply_cmap_to_tensor(tensor : torch.Tensor, cmap : matplotlib.colors.Colormap, norm : matplotlib.colors.Normalize = None):
    tensor = tensor.cpu().numpy()
    if norm is not None:
        tensor = norm(tensor)
    tensor = cmap(tensor) # it converts tensor to numpy rgba image with hwc format, .astype(np.float32)
    tensor = torch.from_numpy(tensor)
    tensor = tensor[:,:,:3] # delete alpha channel
    tensor = tensor.permute(2, 0, 1) # hwc -> chw
    
    return tensor

def make_grid_image_extended(mode : str, image : torch.Tensor, label : torch.Tensor, prediction : torch.Tensor, label_map : dict, idx):
        bone_cmap = plt.get_cmap("bone")

        whs_label_map = dict(list(label_map.items())[:8])
        whs_cmap, whs_norm = create_custom_cmap_extended(whs_label_map)

        heart_label_map = dict(list(label_map.items())[:1] + list(label_map.items())[8:])
        heart_cmap, heart_norm = create_custom_cmap_extended(heart_label_map)

        image = image.squeeze().cpu() # delete batch
        image = apply_cmap_to_tensor(image[:,:,idx].T, bone_cmap)
        
        whs_label = label.squeeze().cpu() # delete batch
        whs_label = torch.argmax(whs_label[:8], dim=0)
        whs_label = apply_cmap_to_tensor(whs_label[:,:,idx].T, whs_cmap, whs_norm)
        
        heart_label = torch.concat((label[:1], label[8:])).squeeze().cpu() # delete batch
        heart_label = torch.argmax(heart_label, dim=0)
        heart_label = apply_cmap_to_tensor(heart_label[:,:,idx].T, heart_cmap, heart_norm)
        
        whs_pred = prediction[:8].squeeze().float().cpu() # delete batch
        whs_pred = torch.argmax(whs_pred, dim=0)
        whs_pred = apply_cmap_to_tensor(whs_pred[:,:,idx].T, whs_cmap, whs_norm)

        heart_pred = torch.concat((prediction[:1], prediction[8:])).squeeze().float().cpu() # delete batch
        heart_pred = torch.argmax(heart_pred, dim=0)
        heart_pred = apply_cmap_to_tensor(heart_pred[:,:,idx].T, heart_cmap, heart_norm)
        
        img_grid = torchvision.utils.make_grid([image, whs_label, heart_label, whs_pred, heart_pred])
        if mode == "tensorboard":
            return img_grid
        if mode == "eval":
            img_grid = Image.fromarray((img_grid.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        
        return img_grid

def postprocess_outputs_extended(outputs: torch.Tensor, label_values, indices: list):
    selected_outputs = torch.cat([outputs[i].unsqueeze(0) for i in indices], dim=0)
    segmented = torch.softmax(selected_outputs, dim=0).argmax(dim=0)  # 128, 128, 128
    
    return (segmented[None, ...] == torch.tensor(label_values, device=segmented.device)[:, None, None, None]).int()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_extended(model, image_dir, label_dir, dataset_config, checkpoint, output_dir, tag=""):
    dataset = load_config(dataset_config)
    print("train_size: {}, val_size: {}, test_size: {}".format(len(dataset["train"]), len(dataset["val"]), len(dataset["test"])))
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotExtendedd(keys=["label"]),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),    
    ])
    
    test_dataset = Dataset(dataset["test"], transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # model
    print("Load Model:", model)
    
    model = UNETR(
        in_channels=1,
        out_channels=9,
        img_size=(128, 128, 128),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    print("Load Checkpoint:", os.path.basename(checkpoint))
    model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    label_names = list(dataset["test"][0]["label_map"].keys())
    table = pd.DataFrame({"idx": [], **{label_name: float for label_name in label_names}, "mean": []})
    with torch.no_grad():
        model.eval()
        for idx, batch in tqdm(zip(dataset["test"], test_dataloader)):
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            
            outputs = outputs.as_tensor()
            targets = targets.as_tensor()
            outputs = outputs.squeeze()
            targets = targets.squeeze().int()
            
            whs_outputs = postprocess_outputs_extended(outputs, list(range(8)), indices=list(range(8)))
            heart_outputs = postprocess_outputs_extended(outputs, [0.0, 1.0], indices=[0, 8])[1].unsqueeze(0)
            processed_outputs = torch.concat((whs_outputs, heart_outputs))
            
            metrics = Metrics(processed_outputs, targets)
            mean_dice = metrics.meanDice().item()
            dice_by_classes = metrics.Dice().cpu().numpy()
            if output_dir:
                img_grid = make_grid_image_extended("eval", inputs, targets, processed_outputs, batch["label_map"], 50)
                img_grid.save(os.path.join(output_dir, tag, "{}_img_grid.png".format(os.path.basename(idx["image"]))))
                new_row = pd.DataFrame([[os.path.basename(idx["image"]), *dice_by_classes, mean_dice]], columns=table.columns)
                table = pd.concat([table, new_row])
    # print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="center", floatfmt=".4f"))
    table = table.round(4)
    print(table)
    if output_dir:
        print("Save Table to", os.path.join(output_dir, tag, "result.csv"))
        table.to_csv(os.path.join(output_dir, tag, "result.csv"), index=False)