import torch
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

from heart_seg_app.utils.config import save_config, load_config
from heart_seg_app.utils.dataset import (
    ToOneHotd,
    collect_data_paths, split_dataset, label_postprocessing)
from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.utils.visualization import make_grid_image
from heart_seg_app.models.unetr import unetr

from monai.data.dataset import Dataset
from monai.data import DataLoader, NumpyReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, 
    RandFlipd, RandAffined, RandGaussianNoised, RandGaussianSmoothd, 
    NormalizeIntensityd, RandAdjustContrastd, Rand3DElasticd,
)
from monai.losses import DiceLoss
from monai.utils import set_determinism, get_seed

import os
from tqdm import tqdm

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

hyperparams = {
    "model": "UNETR",
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
        }
    },
    "loss_function": "DiceLoss",
    "epochs": 0,
    "batch_size": 1,
    "seed": None,
}

def train(model, image_dir, label_dir, dataset_config=None, split_ratios=(0.75, 0.2, 0.05), seed=0, tag="", checkpoint=None, output_dir=None, epochs=3):
    set_determinism(seed=seed) if seed is not None else get_seed()
    hyperparams["seed"] = seed
    hyperparams["epochs"] = epochs
    
    data = collect_data_paths(image_dir, label_dir, postfix=".gz.128128128.npy")
    print(f"Dataset Size: {len(data)}")
    
    if dataset_config:
        dataset = load_config(dataset_config)
    else:
        dataset = split_dataset(data, split_ratios=split_ratios, seed=seed)

    print("train_size: {}, val_size: {}, test_size: {}".format(len(dataset["train"]), len(dataset["val"]), len(dataset["test"])))
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotd(keys=["label"], label_values=label_values),
        EnsureTyped(keys=["image", "label"]),
        
        # Spacing
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandAffined(
            keys=["image", "label"], 
            prob=0.7,
            rotate_range=(0.1, 0.1, 0.1), 
            scale_range=(0.1, 0.1, 0.1), 
            translate_range=(5, 5, 5), 
            mode=("bilinear", "nearest")
        ),
        Rand3DElasticd(keys=["image", "label"], prob=0.2, sigma_range=(5, 8), magnitude_range=(100, 200)),

        # Intensity
        NormalizeIntensityd(keys=["image"], channel_wise=True),  # (data - mean) / std
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.5)),

        # Crop
        # RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(128, 128, 128), num_samples=2), 
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotd(keys=["label"], label_values=label_values),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),    
    ])

    train_dataset = Dataset(dataset["train"], transform=train_transforms)
    val_dataset = Dataset(dataset["val"], transform=val_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    
    # model
    if model == "unetr":
        model = unetr()
        hyperparams["model"] = "UNETR"
    if checkpoint:
        print("Load Checkpoint:", os.path.basename(checkpoint))
        model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), **(hyperparams["optimizer"]["params"]))
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Save run's path: ", os.path.abspath(output_dir))
            
        
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
        writer = SummaryWriter(os.path.join(output_dir, tag))
    
        dataset["label_color_map"] = label_color_map
        print("Save Dataset Config: ", os.path.join(output_dir, tag, "dataset.json"))
        print("Save Hyperparams Config: ", os.path.join(output_dir, tag, "hyperparams.json"))
        save_config(dataset, os.path.join(output_dir, tag, "dataset.json"))
        save_config(hyperparams, os.path.join(output_dir, tag, "hyperparams.json"))
        
    # training step
    best_val_mean_dice = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mean_dice = 0
        train_mean_dice_by_classes = torch.zeros(len(label_values), device=device)
        for batch in tqdm(train_dataloader):
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            
            outputs = label_postprocessing(outputs)
            targets = targets.int().squeeze(0)
            
            metrics = Metrics(outputs, targets)
            mean_dice = metrics.meanDice().item()
            dice_by_classes = metrics.Dice()
            train_mean_dice += mean_dice
            train_mean_dice_by_classes += dice_by_classes
            
        train_loss /= len(train_dataloader)
        train_mean_dice /= len(train_dataloader)
        train_mean_dice_by_classes /= len(train_dataloader)
        
        # validation step
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_mean_dice = 0
            val_mean_dice_by_classes = torch.zeros(len(label_values), device=device)
            for batch in tqdm(val_dataloader):
                inputs, targets = batch["image"].to(device), batch["label"].to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                
                outputs = label_postprocessing(outputs)
                targets = targets.int().squeeze(0)
                
                metrics = Metrics(outputs, targets)
                mean_dice = metrics.meanDice().item()
                dice_by_classes = metrics.Dice()
                val_mean_dice += mean_dice
                val_mean_dice_by_classes += dice_by_classes
            val_mean_dice /= len(val_dataloader)
            val_mean_dice_by_classes /= len(val_dataloader)
            print(f"{epoch+1}/{epochs}: train_loss={train_loss:.5}, train_mean_dice={train_mean_dice:.5}",
                  "by classes: ", [f"{elem:.5}" for elem in train_mean_dice_by_classes])
            print(f"{epoch+1}/{epochs}: val_loss={val_loss:.5}, val_mean_dice={val_mean_dice:.5}",
                  "by classes: ", [f"{elem:.5}" for elem in val_mean_dice_by_classes])
            
            if output_dir:
                if val_mean_dice > best_val_mean_dice:
                    best_val_mean_dice = val_mean_dice
                    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"{tag}.pth"))
                else:
                    print(f"No improvement in val mean dice, skip saving checkpoint\n best_val_mean_dice={best_val_mean_dice:.5}")
                
                writer.add_scalar("train_loss", train_loss, epoch+1)
                writer.add_scalar("train_mean_dice", train_mean_dice, epoch+1)
                writer.add_scalar("val_loss", val_loss, epoch+1)
                writer.add_scalar("val_mean_dice", val_mean_dice, epoch+1)
                class_names = [value[1] for value in label_color_map.values()]
                writer.add_scalars("train_mean_dice_by_classes", {class_names[i]: train_mean_dice_by_classes[i] for i in range(len(label_color_map))}, epoch+1)
                writer.add_scalars("val_mean_dice_by_classes", {class_names[i]: val_mean_dice_by_classes[i] for i in range(len(label_color_map))}, epoch+1)
                img_grid = make_grid_image("tensorboard", inputs, targets, outputs, label_color_map, 50)
                writer.add_image("validation_grid", img_grid, epoch+1)