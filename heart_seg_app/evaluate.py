import os

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, NumpyReader
from monai.data.dataset import Dataset
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Spacingd,
    NormalizeIntensityd,
    MapTransform,
)
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
from torchvision.utils import make_grid

from heart_seg_app.utils.config import load_config
from heart_seg_app.utils.metrics import Dice
from heart_seg_app.utils.visualization import create_custom_cmap


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
CLASS_NAMES = ["background", "left ventricle", "right ventricle", "ascending aorta"]
N_CLASSES = 4

def get_model(model : str, n_classes : int) -> torch.nn.Module:
    if model == "unetr":
        hyperparams["model"] = "UNETR"
        return UNETR(
            in_channels=1,
            out_channels=n_classes,
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

def collect_data(image_dir : str, label_dir : str) -> list[dict[str, str]]:
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))
    
    data = []
    for image, label in zip(images, labels):
        data.append({"image": os.path.join(image_dir, image),
                     "label": os.path.join(label_dir, label)})
        
    return data

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

def validate(
    model : torch.nn.Module,
    batch : dict[str, torch.Tensor],
    device : torch.device,
    n_classes : int,
):
    model.eval()
    step_dice = 0
    step_dice_by_classes = torch.zeros(n_classes, device=device)
    dice = Dice()
    
    with torch.no_grad():
        inputs, targets = batch["image"].to(device), batch["label"].to(device)

        outputs = model(inputs); outputs : torch.Tensor
        
        outputs = postprocess_outputs(outputs, n_classes).int()
        targets = targets.int()
        
        dice_by_classes = dice(outputs, targets)
        mean_dice = dice.mean().item()
        step_dice += mean_dice
        step_dice_by_classes += dice_by_classes

        # visualization
        targets = torch.argmax(targets, dim=1).squeeze().cpu()
        outputs = torch.argmax(outputs, dim=1).squeeze().cpu()
        colors = ["black", "yellow", "skyblue", "orange"]
        label_values = list(range(4))
        cmap, norm = create_custom_cmap(label_values, colors)            
        img_grid = make_grid_image([inputs, targets, outputs],
                                    [plt.get_cmap("bone"), cmap, cmap],
                                    [None, norm, norm],
                                    50)
        img_grid = Image.fromarray((img_grid.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        
    return step_dice, step_dice_by_classes, img_grid

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

def make_grid_image(images : list[torch.Tensor],
                    cmaps : list[matplotlib.colors.Colormap],
                    norms : list[matplotlib.colors.Normalize],
                    idx : int):
    for i, image in enumerate(images):
        image = torch.Tensor(image).squeeze().cpu()[:,:,idx].T
        images[i] = apply_cmap_to_tensor(image, cmaps[i], norms[i])

    return make_grid(images)

def evaluate(model : str,
             image_dir : str,
             label_dir : str,
             dataset_config : str,
             tag : str,
             checkpoint : str,
             output_dir : str = None,
             device : torch.device = "cuda" if torch.cuda.is_available() else "cpu",
             ) -> None:
    
    data = collect_data(image_dir, label_dir)
    print(f"Dataset Test Size: {len(data)}")
    dataset = load_config(dataset_config)
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NumpyReader),
        EnsureChannelFirstd(keys=["image"]),
        ToOneHotMetad(keys=["label"],
                      value_extractor=lambda _: list(range(4))),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], channel_wise=True),    
    ])
    
    test_dataset = Dataset(dataset["test"], transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    model = get_model(model, N_CLASSES); model : torch.nn.Module
    print("Load Checkpoint:", checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.to(device)

    results = []
    
    mean_dice = 0
    mean_dice_by_classes = torch.zeros(N_CLASSES)
    for meta, batch in tqdm(zip(dataset["test"], test_dataloader)):
        idx = os.path.basename(meta["image"])
        dice, dice_by_classes, img_grid = validate(model, batch, device, N_CLASSES)
        dice_by_classes = dice_by_classes.cpu().numpy()
        mean_dice += dice
        mean_dice_by_classes += dice_by_classes
        
        results.append({"idx": idx,
                        **dict(zip(CLASS_NAMES, dice_by_classes.tolist())),
                        "mean": dice})
        
        if output_dir:
            img_grid.save(os.path.join(output_dir, tag, "{}_img_grid.png".format(idx)))
        
    mean_dice /= len(test_dataloader)
    mean_dice_by_classes /= len(test_dataloader)
    results.append({"idx": "overall",
                    **dict(zip(CLASS_NAMES, mean_dice_by_classes.tolist())),
                    "mean": mean_dice})
    
    results_df = pd.DataFrame(results)
    print(results_df.dtypes)
    results_df = results_df.round(4)
    
    print(tabulate(results_df, headers="keys", tablefmt="fancy_grid", numalign="center", floatfmt=".4f"))
    
    from heart_seg_app.utils.visualization import create_custom_cmap
    import matplotlib.pyplot as plt

    if output_dir:
        print("Save Table to", os.path.join(output_dir, tag, "results.csv"))
        results_df.to_csv(os.path.join(output_dir, tag, "results.csv"), index=False)