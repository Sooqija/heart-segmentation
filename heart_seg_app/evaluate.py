import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

from heart_seg_app.utils.dataset import Dataset, ImagePreprocessing, LabelPreprocessing, label_postprocessing
from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.utils.visualization import make_grid_image
from heart_seg_app.models.unetr import unetr

import os
import numpy as np
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

def evaluate(model, image_dir, label_dir, checkpoint, output_dir, tag=""):
    if checkpoint is None:
        print("No checkpoint specified")
        return
    if output_dir is None:
        print("No output directory specified")
        return
    
    train_dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transforms.Compose([
            ImagePreprocessing()]),
        target_transform=transforms.Compose(transforms=[
            LabelPreprocessing(label_values),
        ]),
        postfix=".gz.128128128.npy"
    )
    
    print("Dataset size: {}".format(len(train_dataset)))
    
    train_size = int(0.75 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    print(f"train_images: {train_size}, validation_images: {val_size}, test_images {test_size}")

    torch.manual_seed(0)
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size]
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # model
    if model == "unetr":
        model = unetr()
    print("Load Checkpoint:", os.path.basename(checkpoint))
    model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    label_names = [value[1] for value in label_color_map.values()]
    table = [ ["Sample Idx", *label_names, "mean"] ]
    with torch.no_grad():
        model.eval()
        for i, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            outputs = label_postprocessing(outputs)
            targets = targets.squeeze(0).int()
            
            metrics = Metrics(outputs, targets)
            mean_dice = metrics.meanDice().item()
            dice_by_classes = metrics.Dice()
            table.append([i, *dice_by_classes, mean_dice])
            img_grid = make_grid_image("eval", inputs, targets, outputs, label_color_map,50)
            img_grid.save(os.path.join(output_dir, tag, f"{i}_img_grid.png"))
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="center", floatfmt=".4f"))