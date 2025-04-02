import os
import torch
from tqdm import tqdm

import matplotlib.colors
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.utils.config import save_config, load_config

from monai.data.dataset import Dataset
from monai.data import DataLoader, NumpyReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, 
    RandFlipd, RandAffined, RandGaussianNoised, RandGaussianSmoothd, 
    NormalizeIntensityd, RandAdjustContrastd, Rand3DElasticd,
    MapTransform
)
from monai.losses import DiceLoss
from monai.utils import set_determinism, get_seed
from monai.networks.nets import UNETR

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

class PerChannelDiceLoss(torch.nn.Module):
    def __init__ (self, **kwargs):
        super().__init__()
        self.dice_loss = DiceLoss(**kwargs)
        
    def forward(self, predictions, targets, mask):
        channels = predictions.shape[1]
        
        losses = []
        
        for c in range(channels):
            if mask[c]:
                pred_c = predictions[:, c:c+1, ...]
                target_c = targets[:, c:c+1, ...]
                
                loss_c = self.dice_loss(pred_c, target_c)
                losses.append(loss_c)
                
        return sum(losses) / len(losses)

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
        
        img_grid = make_grid([image, whs_label, heart_label, whs_pred, heart_pred])
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

def train_extended(model, image_dir, label_dir, dataset_config=None, split_ratios=(0.75, 0.2, 0.05), seed=0, tag="", checkpoint=None, output_dir=None, epochs=3):
    set_determinism(seed=seed) if seed is not None else get_seed()
    hyperparams["seed"] = seed
    hyperparams["epochs"] = epochs
    
    dataset = load_config(dataset_config)
    print("train_size: {}, val_size: {}, test_size: {}".format(len(dataset["train"]), len(dataset["val"]), len(dataset["test"])))
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotExtendedd(keys=["label"]),
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
        ToOneHotExtendedd(keys=["label"]),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),    
    ])
    
    train_dataset = Dataset(dataset["train"], transform=train_transforms)
    val_dataset = Dataset(dataset["val"], transform=val_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    
    # model
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
    hyperparams["model"] = "UNETR"
    if checkpoint:
        print("Load Checkpoint:", os.path.basename(checkpoint))
        model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    loss_function = PerChannelDiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), **(hyperparams["optimizer"]["params"]))
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Save run's path: ", os.path.abspath(output_dir))
            
        
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
        writer = SummaryWriter(os.path.join(output_dir, tag))
    
        print("Save Dataset Config: ", os.path.join(output_dir, tag, "dataset.json"))
        print("Save Hyperparams Config: ", os.path.join(output_dir, tag, "hyperparams.json"))
        save_config(dataset, os.path.join(output_dir, tag, "dataset.json"))
        save_config(hyperparams, os.path.join(output_dir, tag, "hyperparams.json"))
        
    best_val_mean_dice = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mean_dice = 0
        train_mean_dice_by_classes = torch.zeros(9, device=device)
        for batch in tqdm(train_dataloader):
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss_mask = create_active_classes_mask(batch["label_map"])
            loss = loss_function(outputs, targets, mask=loss_mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            
            outputs = outputs.as_tensor()
            targets = targets.as_tensor()
            outputs = outputs.squeeze()
            targets = targets.squeeze().int()

            whs_outputs = postprocess_outputs_extended(outputs, list(range(8)), indices=list(range(8)))
            heart_outputs = postprocess_outputs_extended(outputs, [0.0, 1.0], indices=[0, 8])[1].unsqueeze(0)
            processed_outputs = torch.concat((whs_outputs, heart_outputs))
            
            metrics = Metrics(processed_outputs, targets)
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
            val_mean_dice_by_classes = torch.zeros(9, device=device)
            for batch in tqdm(val_dataloader):
                inputs, targets = batch["image"].to(device), batch["label"].to(device)
                outputs = model(inputs)
                loss_mask = create_active_classes_mask(batch["label_map"])
                loss = loss_function(outputs, targets, mask=loss_mask)  
                val_loss += loss.item()
                
                outputs = outputs.as_tensor()
                targets = targets.as_tensor()
                outputs = outputs.squeeze()
                targets = targets.squeeze().int()
            
                whs_outputs = postprocess_outputs_extended(outputs, list(range(8)), indices=list(range(8)))
                heart_outputs = postprocess_outputs_extended(outputs, [0.0, 1.0], indices=[0, 8])[1].unsqueeze(0)
                processed_outputs = torch.concat((whs_outputs, heart_outputs))
                
                metrics = Metrics(processed_outputs, targets)
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
                class_names = list(batch["label_map"].keys())
                writer.add_scalars("train_mean_dice_by_classes", {class_names[i]: train_mean_dice_by_classes[i] for i in range(9)}, epoch+1)
                writer.add_scalars("val_mean_dice_by_classes", {class_names[i]: val_mean_dice_by_classes[i] for i in range(9)}, epoch+1)
                img_grid = make_grid_image_extended("tensorboard", inputs, targets, processed_outputs, batch["label_map"], 50)
                writer.add_image("validation_grid", img_grid, epoch+1)