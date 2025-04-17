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
from heart_seg_app.utils.metrics import Dice

from monai.data.dataset import Dataset
from monai.networks.nets import UNETR
from monai.data import DataLoader, NumpyReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, 
    RandFlipd, RandAffined, RandGaussianNoised, RandGaussianSmoothd, 
    NormalizeIntensityd, RandAdjustContrastd, Rand3DElasticd,
    MapTransform,
)
from monai.losses import DiceLoss
from monai.utils import set_determinism, get_seed

import os
from tqdm import tqdm

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

def collect_data(image_dir : str, label_dir : str) -> list[dict[str, str]]:
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))
    
    data = []
    for image, label in zip(images, labels):
        data.append({"image": os.path.join(image_dir, image),
                     "label": os.path.join(label_dir, label)})
        
    return data

def get_model(model : str) -> torch.nn.Module:
    if model == "unetr":
        hyperparams["model"] = "UNETR"
        return UNETR(
            in_channels=1,
            out_channels=4,
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

class ToOneHotMetad(MapTransform):
    def __init__(self, keys : list[str], value_extractor : callable):
        super().__init__(keys)
        self.value_extractor = value_extractor
        
    def __call__(self, data):
        d = dict(data); d : dict[str, torch.Tensor]
        label_values = self.value_extractor(d)
        for key in self.keys:
            d[key] = (d[key] == torch.tensor(label_values)[:, None, None, None]).int()
            
        return d

def postprocess_outputs(outputs : torch.Tensor, n_classes : int) -> torch.Tensor:
    segmented = torch.softmax(outputs, dim=1).argmax(dim=1).unsqueeze(dim=1)
    multilabel_mask = torch.tensor(range(n_classes), device=segmented.device)[None, :, None, None, None]
    return (segmented == multilabel_mask).float()

def train_step(
    model : torch.nn.Module,
    dataloader : DataLoader,
    optimizer : torch.optim.Optimizer,
    loss_function : torch.nn.Module,
    device : torch.device,
    n_classes : int,
):
    model.train()
    step_loss = 0
    step_dice = 0
    step_dice_by_classes = torch.zeros(n_classes, device=device)
    dice = Dice()
    
    for batch in tqdm(dataloader):
        batch : dict[str, torch.Tensor]
        inputs, targets = batch["image"].to(device), batch["label"].to(device)

        outputs = model(inputs)
        loss = loss_function(outputs, targets); loss : torch.Tensor
        loss.backward()
        step_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        
        outputs = postprocess_outputs(outputs, 4).int()
        targets = targets.int()
        
        dice_by_classes = dice(outputs, targets)
        mean_dice = dice.mean().item()
        step_dice += mean_dice
        step_dice_by_classes += dice_by_classes
        
        
    step_loss /= len(dataloader)
    step_dice /= len(dataloader)
    step_dice_by_classes /= len(dataloader)
    
    return step_loss, step_dice, step_dice_by_classes

def validation_step(
    model : torch.nn.Module,
    dataloader : DataLoader,
    loss_function : torch.nn.Module,
    device : torch.device,
    n_classes : int,
):
    model.eval()
    step_loss = 0
    step_dice = 0
    step_dice_by_classes = torch.zeros(n_classes, device=device)
    dice = Dice()
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch : dict[str, torch.Tensor]
            inputs, targets = batch["image"].to(device), batch["label"].to(device)

            outputs = model(inputs); outputs : torch.Tensor
            loss = loss_function(outputs, targets); loss : torch.Tensor
            step_loss += loss.item()
            
            outputs = postprocess_outputs(outputs, 4).int()
            targets = targets.int()
            
            dice_by_classes = dice(outputs, targets)
            mean_dice = dice.mean().item()
            step_dice += mean_dice
            step_dice_by_classes += dice_by_classes
    
            last_batch = batch
            last_batch["pred"] = outputs
        
    step_loss /= len(dataloader)
    step_dice /= len(dataloader)
    step_dice_by_classes /= len(dataloader)
    
    return step_loss, step_dice, step_dice_by_classes, last_batch

def write_logs(
    writer : SummaryWriter,
    train_loss : float,
    train_dice : float,
    train_dice_by_classes : torch.Tensor,
    val_loss : float,
    val_dice : float,
    val_dice_by_classes : torch.Tensor,
    class_names : list[str],
    step : int,
):
    # train
    writer.add_scalar("train/loss", train_loss, step)
    writer.add_scalar("train/mean_dice", train_dice, step)
    writer.add_scalars("train/mean_dice_by_classes",
                       dict(zip(class_names, train_dice_by_classes)),
                       step)
    
    # validation
    writer.add_scalar("val/loss", val_loss, step)
    writer.add_scalar("val/mean_dice", val_dice, step)
    writer.add_scalars("val/mean_dice_by_classes",
                       dict(zip(class_names, val_dice_by_classes)),
                       step)
    
    # compare
    writer.add_scalars("compare/loss", {"train": train_loss, "val": val_loss}, step)
    writer.add_scalars("compare/mean_dice", {"train": train_dice, "val": val_dice}, step)

import matplotlib.colors    
def apply_cmap_to_tensor(tensor : torch.Tensor,
                         cmap : matplotlib.colors.Colormap,
                         norm : matplotlib.colors.Normalize = None
                         ) -> torch.Tensor:
    tensor = tensor.cpu().numpy()
    if norm is not None:
        tensor = norm(tensor)
    tensor = cmap(tensor) # it converts tensor to numpy rgba image with hwc format, .astype(np.float32)
    tensor = torch.from_numpy(tensor)
    tensor = tensor[:,:,:3] # delete alpha channel
    tensor = tensor.permute(2, 0, 1) # hwc -> chw

    return tensor

from torchvision.utils import make_grid
import matplotlib.cm
import matplotlib.pyplot as plt
from heart_seg_app.utils.visualization import create_custom_cmap
def make_grid_image(images : list[torch.Tensor],
                    cmaps : list[matplotlib.colors.Colormap],
                    norms : list[matplotlib.colors.Normalize],
                    idx : int):
    for i, image in enumerate(images):
        image = torch.Tensor(image).squeeze().cpu()[:,:,idx].T
        images[i] = apply_cmap_to_tensor(image, cmaps[i], norms[i])

    return make_grid(images)

def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotMetad(keys=["label"],
                      value_extractor=lambda _: list(range(4))),
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
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotMetad(keys=["label"],
                      value_extractor=lambda _: list(range(4))),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),    
    ])
    
    return train_transforms, val_transforms
          
def train(model : str,
          image_dir : str,
          label_dir : str,
          dataset_config : str = None,
          split_ratios : tuple[float, float, float] = (0.75, 0.2, 0.05),
          seed : int = None,
          tag : str = "",
          checkpoint : str = None,
          output_dir : str = None,
          epochs : int = 3,
          batch_size : int = 1,
          lr : float = 1e-4,
          device : torch.device = "cuda" if torch.cuda.is_available() else "cpu"
          ) -> None:
    seed = seed if seed is not None else get_seed()
    set_determinism(seed=seed)
    # hyperparams["seed"] = seed
    # hyperparams["epochs"] = epochs

    data = collect_data(image_dir, label_dir)
    print(f"Dataset Size: {len(data)}")
    if dataset_config:
        dataset = load_config(dataset_config)
    else:
        dataset = split_dataset(data, split_ratios=split_ratios, seed=seed)
    
    print("train_size: {}, val_size: {}, test_size: {}".format(len(dataset["train"]), len(dataset["val"]), len(dataset["test"])))
    
    train_transforms, val_transforms = get_transforms()

    train_dataset = Dataset(dataset["train"], transform=train_transforms)
    val_dataset = Dataset(dataset["val"], transform=val_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # model
    model = get_model(model); model : torch.nn.Module
    if checkpoint:
        print("Load Checkpoint:", os.path.basename(checkpoint))
        model.load_state_dict(torch.load(os.path.join(checkpoint), weights_only=True))
    model = model.to(device)
    
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), **(hyperparams["optimizer"]["params"]))
    
    if output_dir:
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        print("Save run's path: ", os.path.abspath(output_dir))
        
        writer = SummaryWriter(os.path.join(output_dir, tag))
    
        save_config(dataset, os.path.join(output_dir, tag, "dataset.json"))
        print("Save Dataset Config: ", os.path.join(output_dir, tag, "dataset.json"))
        save_config(hyperparams, os.path.join(output_dir, tag, "hyperparams.json"))
        print("Save Hyperparams Config: ", os.path.join(output_dir, tag, "hyperparams.json"))
    
    best_dice = 0
    for epoch in range(epochs):
        # train step
        train_loss, train_dice, train_dice_by_classes = train_step(
            model, 
            train_dataloader, 
            optimizer,
            loss_function, 
            device,
            4,
        )
        
        # validation step
        val_loss, val_dice, val_dice_by_classes, last_batch = validation_step(
            model,
            val_dataloader,
            loss_function,
            device,
            4,
        )
        
        print(f"{epoch+1}/{epochs}: train_loss={train_loss:.5}, train_mean_dice={train_dice:.5}",
                  "by classes: ", [f"{elem:.5}" for elem in train_dice_by_classes])
        print(f"{epoch+1}/{epochs}: val_loss={val_loss:.5}, val_mean_dice={val_dice:.5}",
                  "by classes: ", [f"{elem:.5}" for elem in val_dice_by_classes])
        
        if output_dir:
            if best_dice < val_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", f"{tag}.pth"))
            else:
                print(f"No improvement in val mean dice, skip saving checkpoint \
                      best_val_mean_dice={best_dice:.5}")
            # TODO: do more dynamic
            class_names = ["background", "left ventricle", "right ventricle", "ascending aorta"]
            write_logs(writer, train_loss, train_dice, train_dice_by_classes, val_loss, val_dice, val_dice_by_classes, class_names, epoch+1)
            
            # visualize last batch
            image, label, pred = last_batch["image"][0], last_batch["label"][0], last_batch["pred"][0]
            
            image = torch.squeeze(image, dim=0)
            label = torch.argmax(label, dim=0)
            pred = torch.argmax(pred, dim=0)

            colors = ["black", "yellow", "skyblue", "orange"]
            label_values = list(range(4))
            cmap, norm = create_custom_cmap(label_values, colors)
            
            img_grid = make_grid_image([image, label, pred],
                                       [plt.get_cmap("bone"), cmap, cmap],
                                       [None, norm, norm],
                                       50)
            writer.add_image("val/image_grid", img_grid, epoch+1)
    writer.close()
    print(f"Training complete. Best validation Dice: {best_dice:.4f}")