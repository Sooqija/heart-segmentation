import argparse
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
import mmwhs_metrics
import mmwhs_model
import visualization

parser = argparse.ArgumentParser(description='Heart Segmentation App')

parser.add_argument('--mode', choices=['train', 'eval', 'vis'], help='Mode of operation', required=True)
parser.add_argument('--model_path', action='store', help='Path to your model', default='./checkpoints/mmwhs_v1.ckpt')
parser.add_argument('--data_path', action='store', help='Path to your data', default='./data/train')
parser.add_argument('--device', action='store', help='Device', default=None)






label_maps = {
    0.0: "background",
    205.0: "the myocardium of the left ventricle",
    420.0: "the left atrium blood cavity",
    500.0: "the left ventricle blood cavity",
    550.0: "the right atrium blood cavity",
    600.0: "the right ventricle blood cavity",
    820.0: "the ascending aorta",
    850.0: "the pulmonary artery",
}
args : argparse.Namespace = parser.parse_args()
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path :str = args.model_path
data_path = args.data_path



if args.mode == 'train':
    pass
elif args.mode == 'eval':
    train_dataset = dataset.Dataset(
        os.path.join(data_path),
        os.path.join(data_path),
        transform=transforms.Compose([transforms.ToTensor(),
                                    dataset.ImagePreprocessing()]),
        target_transform=transforms.Compose(transforms=[transforms.ToTensor(),
                                                        dataset.LabelPreprocessing(list(label_maps.keys()))]),
        postfix=".256256128.npy"
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    hparams = {
        'n_channels':1,
        'n_classes':8,
    }
    model = mmwhs_model.Unet3D(None, train_loader, hparams)

    if model_path.endswith('.ckpt'):
        checkpoint_file = torch.load('checkpoints/mmwhs_v1.ckpt', weights_only=False)
        model.load_state_dict(checkpoint_file['state_dict'])
    elif model_path.endswith('.pth'):
        model.load_state_dict(torch.load('checkpoints/mmwhs_v1.pth', weights_only=False))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for inputs, targets in train_loader:
            print("Process next image...")
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = dataset.label_postprocessing(outputs)
            targets = targets.int().squeeze(0).to("cpu")
            metrics = mmwhs_metrics.Metrics(outputs, targets)
            iou = metrics.IoU().numpy()
            mean_iou = metrics.meanIoU().numpy()
            dice = metrics.Dice().numpy()
            mean_dice = metrics.meanDice().numpy()
            for i, (iou_val, dice_val) in enumerate(zip(iou, dice)):
                print(f"IoU class {i+1}: {iou_val:.4f}")
                print(f"Dice class {i+1}: {dice_val:.4f}")
            print(f"meanIoU: {mean_iou:.4f}")
            print(f"meanDice: {mean_dice:.4f}")
            print()
elif args.mode == 'vis':
    train_dataset = dataset.Dataset(
        os.path.join(data_path),
        os.path.join(data_path),
        transform=transforms.Compose([transforms.ToTensor(),
                                    dataset.ImagePreprocessing()]),
        target_transform=transforms.Compose(transforms=[transforms.ToTensor(),
                                                        dataset.LabelPreprocessing(list(label_maps.keys()))]),
        postfix=".256256128.npy"
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    hparams = {
        'n_channels':1,
        'n_classes':8,
    }
    model = mmwhs_model.Unet3D(None, train_loader, hparams)

    if model_path.endswith('.ckpt'):
        checkpoint_file = torch.load('checkpoints/mmwhs_v1.ckpt', weights_only=False)
        model.load_state_dict(checkpoint_file['state_dict'])
    elif model_path.endswith('.pth'):
        model.load_state_dict(torch.load('checkpoints/mmwhs_v1.pth', weights_only=False))
    model = model.to(device)

    color_dict = {
               "black": [0, "background"],
               "yellow": [1, "the myocardium of the left ventricle"],
               "skyblue": [2, "the left atrium blood cavity"],
               "red": [3, "the left ventricle blood cavity"],
               "purple": [4, "the right atrium blood cavity"],
               "blue": [5, "the right ventricle blood cavity"],
               "orange": [6, "the ascending aorta"],
               "green": [7, "the pulmonary artery"],
    }
    with torch.no_grad():
        model.eval()
        z = next(iter(train_loader))
        z[0] = z[0].to(device)
        z_pred = model(z[0]).cpu()
        z_pred = torch.softmax(z_pred, dim=1).argmax(dim=1).squeeze(0).numpy()
        z_tar = torch.softmax(z[1], dim=1).argmax(dim=1).squeeze(0).numpy()
    visualization.vtk_visualize_3d_numpy_array(z_pred, color_dict)
    visualization.vtk_visualize_3d_numpy_array(z_tar, color_dict)
