import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

torch.backends.cudnn.benchmark = True

from heart_seg_app.utils.dataset import Dataset, ImagePreprocessing, LabelPreprocessing, label_postprocessing
from heart_seg_app.utils.metrics import Metrics
from heart_seg_app.models.unetr import unetr

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


def train(model, image_dir, label_dir, checkpoint=None, epochs=3):
    
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
    
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    print(f"train_images: {train_size}, validation_images: {val_size}, test_images {test_size}")

    torch.manual_seed(0)
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size]
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    if model == "unetr":
        model = unetr()
    model = model.to(device)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # train
    for epoch in range(epochs):
        model.train()
        mean_loss = 0
        mean_dice = 0
        for batch in tqdm(train_dataloader):
            x, y = batch[0].cuda(), batch[1].cuda()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            mean_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            outputs = label_postprocessing(outputs)
            targets = y.int().squeeze(0).to("cpu")
            metrics = Metrics(outputs.cpu(), targets.cpu())
            dice = metrics.meanDice().numpy()
            mean_dice += dice
        mean_loss /= len(train_dataloader)
        mean_dice /= len(train_dataloader)
        # torch.save(model.state_dict(), os.path.join("./checkpoints", "best_metric_model.pth"))
        print(f"{epoch}: loss={mean_loss:.5}, dice={mean_dice:.5}")
        
        # # validation
        # with torch.no_grad():
        #     model.eval()
        #     for inputs, targets in tqdm(val_dataloader):
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         outputs = model(inputs)
        #         outputs = label_postprocessing(outputs)
        #         targets = targets.int().squeeze(0).to("cpu")
        #         metrics = Metrics(outputs, targets)
        #         dice = metrics.Dice().numpy()
        #         print(f"Dice: {dice}")
        #         mean_dice = metrics.meanDice().numpy()
        #         print(f"meanDice: {mean_dice}")
        #     print(f"val_loss: {0}, val_dice: {mean_dice}")
                