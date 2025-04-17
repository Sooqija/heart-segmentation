#!/usr/bin/env python3
import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from monai.data import NumpyReader, Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd,
    RandFlipd, RandAffined, RandGaussianNoised, RandGaussianSmoothd, 
    NormalizeIntensityd, RandAdjustContrastd, Rand3DElasticd,
)

from monai.networks.nets import Unet
from monai.losses import DiceLoss

from heart_seg_app.utils.config import load_config
from heart_seg_app.utils.metrics import Dice

# Constants
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs/localization"

def parse_args():
    parser = argparse.ArgumentParser(description="Train localization model")
    parser.add_argument("--data-dir", type=str, help="Dataset directory", required=True)
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for predictions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()

def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureTyped(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
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
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),
    ])
    return train_transforms, val_transforms

def create_model(device : torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> Unet:
    return Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)

def apply_cmap_to_tensor(tensor : torch.Tensor,
                         cmap : matplotlib.colors.Colormap,
                         norm : matplotlib.colors.Normalize = None) -> torch.Tensor:
    tensor = tensor.cpu().numpy()
    if norm is not None:
        tensor = norm(tensor)
    tensor = cmap(tensor)[..., :3]  # Remove alpha channel
    return torch.from_numpy(tensor).permute(2, 0, 1)  # HWC -> CHW

def make_grid_image(image : torch.Tensor, label : torch.Tensor, prediction : torch.Tensor, idx : int = 16):
    image = torch.Tensor(image)
    label = torch.Tensor(label)
    prediction = torch.Tensor(prediction)
    image = apply_cmap_to_tensor(image[:,:,idx].T, plt.get_cmap("bone"))
    label = (apply_cmap_to_tensor(label[:,:,idx].T, plt.get_cmap("grey")) * 255).int()
    prediction = (apply_cmap_to_tensor(prediction[:,:,idx].T, plt.get_cmap("grey")) * 255).int()
    return make_grid([image, label, prediction])

def train_epoch(model : Unet,
                dataloader : DataLoader,
                loss_fn : DiceLoss,
                optimizer : torch.optim.Optimizer,
                dice_metric : Dice,
                device : torch.device,
                threshold : float
                ) -> tuple[float, float]:
    model.train()
    total_loss, total_dice = 0, 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch : dict[str, torch.Tensor]
        inputs, targets = batch["image"].to(device), batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs); outputs : torch.Tensor
        loss = loss_fn(outputs, targets); loss : torch.Tensor
        loss.backward()
        optimizer.step()
        
        preds = (torch.sigmoid(outputs) > threshold).int()
        dice_metric(preds, targets.int())
        
        total_loss += loss.item()
        total_dice += dice_metric.mean().item()
    
    return total_loss / len(dataloader), total_dice / len(dataloader)

def validation_step(model : Unet,
             dataloader : DataLoader,
             loss_fn : DiceLoss,
             dice_metric : Dice,
             device : torch.device,
             threshold : float
            ) -> tuple[float, float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    model.eval()
    total_loss, total_dice = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch : dict[str, torch.Tensor]
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            
            loss = loss_fn(outputs, targets)
            preds = (torch.sigmoid(outputs) > threshold).int()
            dice_metric(preds, targets.int())
            
            total_loss += loss.item()
            total_dice += dice_metric.mean().item()
            
            # Return last batch for visualization
            if batch == len(dataloader) - 1:
                last_batch = (inputs, targets, preds)
    
    return (total_loss / len(dataloader), 
            total_dice / len(dataloader), 
            last_batch)

def main():
    args = parse_args()
    
    # Setup directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_config(os.path.join(args.data_dir, "dataset.json"))
    train_transforms, val_transforms = get_transforms()
    
    # Create datasets and loaders
    train_dataset = Dataset(dataset["train"], transform=train_transforms)
    val_dataset = Dataset(dataset["val"], transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model and training setup
    model = create_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)
    dice_metric = Dice()
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    best_dice = 0
    for epoch in range(args.epochs):
        # Training
        train_loss, train_dice = train_epoch(
            model, train_loader, loss_fn, optimizer, dice_metric, device, args.threshold
        )
        
        # Validation
        val_loss, val_dice, (val_inputs, val_targets, val_preds) = validation_step(
            model, val_loader, loss_fn, dice_metric, device, args.threshold
        )
        
        # Logging
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Dice", {"train": train_dice, "val": val_dice}, epoch)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "localization_best.pth"))
            print(f"No improvement in val mean dice, skip saving checkpoint\n best_val_mean_dice={best_dice:.5}")
        
        # Visualize
        img_grid = make_grid_image(
            val_inputs[0].squeeze(), 
            val_targets[0].squeeze(), 
            val_preds[0].squeeze()
        )
        writer.add_image("Validation/Prediction", img_grid, epoch)
    
    writer.close()
    print(f"Training complete. Best validation Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()