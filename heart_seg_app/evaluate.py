import torch

from monai.data.dataset import Dataset
from monai.data import DataLoader, NumpyReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, NormalizeIntensityd
)

from heart_seg_app.utils.config import load_config
from heart_seg_app.utils.dataset import (
    ToOneHotd,
    collect_data_paths, split_dataset, label_postprocessing)
from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.utils.visualization import make_grid_image
from heart_seg_app.models.unetr import unetr

import os
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_color_map = {
    "black": [0.0, "background"],
    "yellow": [1.0, "left ventricle"],
    "skyblue": [2.0, "right ventricle"],
    "red": [3.0, "left atrium"],
    "purple": [4.0, "right atrium"],
    "blue": [5.0, "myocarium"],
    "orange": [6.0, "aorta"],
    "green": [7.0, "the pulmonary artery"],
}; label_values = [value[0] for value in label_color_map.values()]

def evaluate(model, image_dir, label_dir, dataset_config, checkpoint, output_dir, tag=""):
    data = collect_data_paths(image_dir, label_dir, postfix=".gz.128128128.npy")
    print(f"Dataset Size: {len(data)}")
    
    dataset = load_config(dataset_config)
    
    print("test_size: {}".format(len(dataset["test"])))
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotd(keys=["label"], label_values=label_values),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),    
    ])
    
    test_dataset = Dataset(dataset["test"], transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # model
    print("Load Model:", model)
    if model == "unetr":
        model = unetr()
    print("Load Checkpoint:", os.path.basename(checkpoint))
    model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    label_names = [value[1] for value in label_color_map.values()]
    table = pd.DataFrame({"idx": [], **{label_name: float for label_name in label_names}, "mean": []})
    with torch.no_grad():
        model.eval()
        for idx, batch in tqdm(zip(dataset["test"], test_dataloader)):
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            
            outputs = label_postprocessing(outputs)
            targets = targets.squeeze(0).int()
            
            metrics = Metrics(outputs, targets)
            mean_dice = metrics.meanDice().item()
            dice_by_classes = metrics.Dice().cpu().numpy()
            if output_dir:
                img_grid = make_grid_image("eval", inputs, targets, outputs, label_color_map,50)
                img_grid.save(os.path.join(output_dir, tag, "{}_img_grid.png".format(os.path.basename(idx["image"]))))
                new_row = pd.DataFrame([[os.path.basename(idx["image"]), *dice_by_classes, mean_dice]], columns=table.columns)
                table = pd.concat([table, new_row])
    # print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="center", floatfmt=".4f"))
    table = table.round(4)
    print(table)
    if output_dir:
        print("Save Table to", os.path.join(output_dir, tag, "result.csv"))
        table.to_csv(os.path.join(output_dir, tag, "result.csv"), index=False)