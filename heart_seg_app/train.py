import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

torch.backends.cudnn.benchmark = True

from heart_seg_app.utils.dataset import Dataset, ImagePreprocessing, LabelPreprocessing, label_postprocessing
from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.models.unetr import unetr

import os
import numpy as np
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


def train(model, image_dir, label_dir, tag, checkpoint=None, output_dir=None, epochs=3):
    
    train_dataset = Dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transforms.Compose([
            #transforms.ToTensor(),
            ImagePreprocessing()]),
        target_transform=transforms.Compose(transforms=[
            #transforms.ToTensor(),
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    print("Test Data:")
    for i in test_dataset.indices:
        print(os.path.basename(test_dataset.dataset.inputs_paths[i]))
    
    # model
    
    if model == "unetr":
        model = unetr()
    if checkpoint:
        print("Load Checkpoint:", os.path.basename(checkpoint))
        model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Save run's path: ", os.path.abspath(output_dir))
        
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
    # train
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mean_dice = 0
        train_mean_dice_by_classes = np.zeros(len(label_values))
        for batch in tqdm(train_dataloader):
            x, y = batch[0].cuda(), batch[1].cuda()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            outputs = label_postprocessing(outputs)
            targets = y.int().squeeze(0).to("cpu")
            
            metrics = Metrics(outputs.cpu(), targets.cpu())
            mean_dice = metrics.meanDice().numpy()
            dice_by_classes = metrics.Dice().numpy()
            train_mean_dice += mean_dice
            train_mean_dice_by_classes += dice_by_classes
            
        train_loss /= len(train_dataloader)
        train_mean_dice /= len(train_dataloader)
        train_mean_dice_by_classes /= len(train_dataloader)
        if output_dir:
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"{tag}.pth"))
        
        # validation
        with torch.no_grad():
            model.eval()
            val_mean_dice_by_classes = np.zeros(len(label_values))
            val_mean_dice = 0
            for inputs, targets in tqdm(val_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = label_postprocessing(outputs)
                targets = targets.int().squeeze(0).to("cpu")
                
                metrics = Metrics(outputs, targets)
                mean_dice = metrics.meanDice().numpy()
                dice_by_classes = metrics.Dice().numpy()
                val_mean_dice += mean_dice
                val_mean_dice_by_classes += dice_by_classes
            val_mean_dice /= len(val_dataloader)
            val_mean_dice_by_classes /= len(val_dataloader)
            print(f"{epoch+1}/{epochs}: train_loss={train_loss:.5}, train_mean_dice={train_mean_dice:.5}",
                  "by classes: ", [f"{elem:.5}" for elem in train_mean_dice_by_classes])
            print(f"{epoch+1}/{epochs}: val_mean_dice={val_mean_dice:.5}",
                  "by classes: ", [f"{elem:.5}" for elem in val_mean_dice_by_classes])